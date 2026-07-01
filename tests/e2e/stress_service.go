/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"text/tabwriter"
	"time"
)

const (
	workflowID     = "e2s-stormcast-fcn3"
	maxInFlight    = 10
	requestPayload = `{"start_time":"2024-01-01T00:00:00","num_hours":20,"run_stormcast":true}`
)

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

var (
	serverURL   = flag.String("server_url", "", "Base URL of the PhysicsNeMo Serve service (required)")
	epToken     = flag.String("ep_token", "", "Bearer token for authentication (required)")
	testTimeMin = flag.Int("test_time_min", 30, "Duration in minutes (-1 for unlimited)")
	cadenceSec  = flag.Int("cadence_sec", 30, "Polling cadence in seconds")
)

// ---------------------------------------------------------------------------
// Per-API statistics
// ---------------------------------------------------------------------------

type apiStats struct {
	calls    atomic.Int64
	success  atomic.Int64
	failures atomic.Int64
}

func (s *apiStats) record(ok bool) {
	s.calls.Add(1)
	if ok {
		s.success.Add(1)
	} else {
		s.failures.Add(1)
	}
}

type statsRegistry struct {
	mu    sync.Mutex
	stats map[string]*apiStats
}

func newStatsRegistry() *statsRegistry {
	return &statsRegistry{stats: make(map[string]*apiStats)}
}

func (r *statsRegistry) get(api string) *apiStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	s, ok := r.stats[api]
	if !ok {
		s = &apiStats{}
		r.stats[api] = s
	}
	return s
}

func (r *statsRegistry) print() {
	r.mu.Lock()
	defer r.mu.Unlock()

	fmt.Println()
	fmt.Println("============================================================")
	fmt.Println("  Stress Test Results")
	fmt.Println("============================================================")
	w := tabwriter.NewWriter(os.Stdout, 2, 4, 3, ' ', 0)
	fmt.Fprintln(w, "API\tCalls\tSuccess\tFailed")
	fmt.Fprintln(w, "---\t-----\t-------\t------")
	for api, s := range r.stats {
		fmt.Fprintf(w, "%s\t%d\t%d\t%d\n", api, s.calls.Load(), s.success.Load(), s.failures.Load())
	}
	w.Flush()
	fmt.Println("============================================================")
}

// ---------------------------------------------------------------------------
// Shared run tracker (thread-safe map of run_id → status)
// ---------------------------------------------------------------------------

type runTracker struct {
	mu   sync.RWMutex
	runs map[string]string // run_id → last known status
}

func newRunTracker() *runTracker {
	return &runTracker{runs: make(map[string]string)}
}

func (t *runTracker) add(runID, status string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.runs[runID] = status
}

func (t *runTracker) setStatus(runID, status string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, ok := t.runs[runID]; ok {
		t.runs[runID] = status
	}
}

func (t *runTracker) getStatus(runID string) string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.runs[runID]
}

func (t *runTracker) inFlightCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	n := 0
	for _, s := range t.runs {
		if s == "queued" || s == "running" {
			n++
		}
	}
	return n
}

func (t *runTracker) inFlightIDs() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var ids []string
	for id, s := range t.runs {
		if s == "queued" || s == "running" {
			ids = append(ids, id)
		}
	}
	return ids
}

func (t *runTracker) allIDs() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var ids []string
	for id := range t.runs {
		ids = append(ids, id)
	}
	return ids
}

// completedNotDownloaded returns run IDs with a terminal status that haven't
// been marked as "downloaded" yet.
func (t *runTracker) completedNotDownloaded() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	var ids []string
	for id, s := range t.runs {
		if isTerminal(s) && s != "downloaded" {
			ids = append(ids, id)
		}
	}
	return ids
}

func isTerminal(status string) bool {
	switch status {
	case "completed", "succeeded", "failed", "error", "aborted", "cancelled":
		return true
	}
	return false
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

var httpClient = &http.Client{
	Timeout: 120 * time.Second,
	Transport: &http.Transport{
		TLSClientConfig:     &tls.Config{MinVersion: tls.VersionTLS12},
		TLSHandshakeTimeout: 30 * time.Second,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
	},
}

func authGet(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+*epToken)
	req.Header.Set("Content-Type", "application/json")
	return httpClient.Do(req)
}

func authPost(ctx context.Context, url string, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+*epToken)
	req.Header.Set("Content-Type", "application/json")
	return httpClient.Do(req)
}

func drainAndClose(resp *http.Response) {
	if resp != nil && resp.Body != nil {
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
}

func isHTTPOk(code int) bool {
	return code >= 200 && code < 300
}

// ---------------------------------------------------------------------------
// Health check (pre-flight and periodic)
// ---------------------------------------------------------------------------

func checkHealth(ctx context.Context, stats *statsRegistry) bool {
	st := stats.get("GET /healthz")
	url := *serverURL + "/healthz"
	resp, err := authGet(ctx, url)
	if err != nil {
		fmt.Printf("[healthz] request error: %v\n", err)
		st.record(false)
		return false
	}
	defer drainAndClose(resp)
	ok := isHTTPOk(resp.StatusCode)
	if !ok {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("[healthz] HTTP %d — body: %s\n", resp.StatusCode, string(body))
	}
	st.record(ok)
	return ok
}

// ---------------------------------------------------------------------------
// Submit inference request, return run_id
// ---------------------------------------------------------------------------

func submitInfer(ctx context.Context, stats *statsRegistry) (string, bool) {
	st := stats.get("POST /v1/infer/.../run")
	url := fmt.Sprintf("%s/v1/infer/%s/run", *serverURL, workflowID)
	resp, err := authPost(ctx, url, []byte(requestPayload))
	if err != nil {
		st.record(false)
		return "", false
	}
	defer drainAndClose(resp)

	body, _ := io.ReadAll(resp.Body)
	if !isHTTPOk(resp.StatusCode) {
		st.record(false)
		return "", false
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		st.record(false)
		return "", false
	}
	runID, _ := result["run_id"].(string)
	if runID == "" {
		st.record(false)
		return "", false
	}
	st.record(true)
	return runID, true
}

// ---------------------------------------------------------------------------
// Query run status
// ---------------------------------------------------------------------------

func queryStatus(ctx context.Context, runID string, stats *statsRegistry) (string, bool) {
	st := stats.get("GET /v1/infer/.../.../status")
	url := fmt.Sprintf("%s/v1/infer/%s/%s/status", *serverURL, workflowID, runID)
	resp, err := authGet(ctx, url)
	if err != nil {
		st.record(false)
		return "", false
	}
	defer drainAndClose(resp)

	body, _ := io.ReadAll(resp.Body)
	if !isHTTPOk(resp.StatusCode) {
		st.record(false)
		return "", false
	}

	var result map[string]interface{}
	if err := json.Unmarshal(body, &result); err != nil {
		st.record(false)
		return "", false
	}

	status, _ := result["status"].(string)
	if status == "" {
		// Try nested execution.status
		if exec, ok := result["execution"].(map[string]interface{}); ok {
			status, _ = exec["status"].(string)
		}
	}
	if status == "" {
		status = "unknown"
	}
	st.record(true)
	return strings.ToLower(status), true
}

// ---------------------------------------------------------------------------
// Download results
// ---------------------------------------------------------------------------

func downloadResult(ctx context.Context, runID string, stats *statsRegistry) bool {
	st := stats.get("GET /v1/infer/.../.../results")
	url := fmt.Sprintf("%s/v1/infer/%s/%s/results", *serverURL, workflowID, runID)
	resp, err := authGet(ctx, url)
	if err != nil {
		st.record(false)
		return false
	}
	defer drainAndClose(resp)

	if !isHTTPOk(resp.StatusCode) {
		st.record(false)
		return false
	}

	filename := runID + ".result.zip"
	f, err := os.Create(filename)
	if err != nil {
		st.record(false)
		return false
	}
	_, copyErr := io.Copy(f, resp.Body)
	f.Close()
	os.Remove(filename)

	if copyErr != nil {
		st.record(false)
		return false
	}
	st.record(true)
	return true
}

// ---------------------------------------------------------------------------
// Thread 1: Inference submitter — keep maxInFlight requests in flight
// ---------------------------------------------------------------------------

func thread1_inferenceSubmitter(ctx context.Context, wg *sync.WaitGroup, tracker *runTracker, stats *statsRegistry) {
	defer wg.Done()
	cadence := time.Duration(*cadenceSec) * time.Second / 2

	fillToMax := func() {
		for tracker.inFlightCount() < maxInFlight {
			if ctx.Err() != nil {
				return
			}
			runID, ok := submitInfer(ctx, stats)
			if ok && runID != "" {
				tracker.add(runID, "queued")
				fmt.Printf("[thread-1] Submitted run %s (in-flight: %d)\n", runID, tracker.inFlightCount())
			} else {
				fmt.Println("[thread-1] Failed to submit inference request")
				time.Sleep(2 * time.Second)
				break
			}
		}
	}

	fillToMax()

	ticker := time.NewTicker(cadence)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, id := range tracker.inFlightIDs() {
				if ctx.Err() != nil {
					return
				}
				status, ok := queryStatus(ctx, id, stats)
				if ok {
					tracker.setStatus(id, status)
				}
			}
			fillToMax()
		}
	}
}

// ---------------------------------------------------------------------------
// Thread 2: Periodic status poller for all in-flight runs
// ---------------------------------------------------------------------------

func thread2_statusPoller(ctx context.Context, wg *sync.WaitGroup, tracker *runTracker, stats *statsRegistry) {
	defer wg.Done()
	cadence := time.Duration(*cadenceSec*2) * time.Second

	ticker := time.NewTicker(cadence)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ids := tracker.inFlightIDs()
			fmt.Printf("[thread-2] Polling status for %d in-flight runs\n", len(ids))
			for _, id := range ids {
				if ctx.Err() != nil {
					return
				}
				status, ok := queryStatus(ctx, id, stats)
				if ok {
					tracker.setStatus(id, status)
					fmt.Printf("[thread-2]   %s → %s\n", id, status)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Thread 3: Result downloader — download once per completed run
// ---------------------------------------------------------------------------

func thread3_resultDownloader(ctx context.Context, wg *sync.WaitGroup, tracker *runTracker, stats *statsRegistry) {
	defer wg.Done()
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, id := range tracker.completedNotDownloaded() {
				if ctx.Err() != nil {
					return
				}
				status := tracker.getStatus(id)
				if status == "completed" || status == "succeeded" {
					fmt.Printf("[thread-3] Downloading results for %s\n", id)
					downloadResult(ctx, id, stats)
				}
				tracker.setStatus(id, "downloaded")
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Thread 4: Periodic miscellaneous API calls
// ---------------------------------------------------------------------------

func thread4_miscAPIs(ctx context.Context, wg *sync.WaitGroup, stats *statsRegistry) {
	defer wg.Done()
	cadence := time.Duration(*cadenceSec) * time.Second

	callGet := func(label, url string) {
		st := stats.get(label)
		resp, err := authGet(ctx, url)
		if err != nil {
			st.record(false)
			fmt.Printf("[thread-4] %s — error: %v\n", label, err)
			return
		}
		defer drainAndClose(resp)
		ok := isHTTPOk(resp.StatusCode)
		st.record(ok)
		if !ok {
			fmt.Printf("[thread-4] %s — HTTP %d\n", label, resp.StatusCode)
		}
	}

	ticker := time.NewTicker(cadence)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			callGet("GET /v1/infer/workflows", *serverURL+"/v1/infer/workflows")
			callGet("GET /v1/infer/.../schema", fmt.Sprintf("%s/v1/infer/%s/schema", *serverURL, workflowID))
			callGet("GET /v1/metrics", *serverURL+"/v1/metrics")
			callGet("GET /prometheus/api/v1/query",
				*serverURL+"/prometheus/api/v1/query?query=physicsnemo_serve_gpu_compute_utilization_percent")
		}
	}
}

// ---------------------------------------------------------------------------
// Thread 5: Periodic health check — cancels everything on failure
// ---------------------------------------------------------------------------

func thread5_healthWatchdog(ctx context.Context, cancel context.CancelFunc, wg *sync.WaitGroup, stats *statsRegistry) {
	defer wg.Done()
	cadence := time.Duration(*cadenceSec) * time.Second

	ticker := time.NewTicker(cadence)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !checkHealth(ctx, stats) {
				fmt.Println("\n[thread-5] FATAL: /healthz check failed — stopping test")
				cancel()
				return
			}
		}
	}
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	flag.Parse()

	if *serverURL == "" || *epToken == "" {
		fmt.Fprintln(os.Stderr, "Usage: stress_service --server_url <URL> --ep_token <TOKEN> [--test_time_min N] [--cadence_sec N]")
		os.Exit(1)
	}
	*serverURL = strings.TrimRight(*serverURL, "/")

	fmt.Println("============================================================")
	fmt.Println("  PhysicsNeMo Serve Stress Test")
	fmt.Println("============================================================")
	fmt.Printf("  Server:       %s\n", *serverURL)
	fmt.Printf("  Duration:     %d min", *testTimeMin)
	if *testTimeMin == -1 {
		fmt.Print(" (unlimited — Ctrl-C to stop)")
	}
	fmt.Println()
	fmt.Printf("  Cadence:      %d sec\n", *cadenceSec)
	fmt.Printf("  In-flight:    %d\n", maxInFlight)
	fmt.Printf("  Workflow:     %s\n", workflowID)
	fmt.Println("============================================================")
	fmt.Println()

	stats := newStatsRegistry()

	// Pre-flight health check with retries for flaky cold connections
	const maxPreflightRetries = 3
	fmt.Println("[pre-flight] Checking /healthz ...")
	healthy := false
	for attempt := 1; attempt <= maxPreflightRetries; attempt++ {
		preflight, cancel0 := context.WithTimeout(context.Background(), 30*time.Second)
		if checkHealth(preflight, stats) {
			cancel0()
			healthy = true
			break
		}
		cancel0()
		if attempt < maxPreflightRetries {
			fmt.Printf("[pre-flight] Attempt %d/%d failed, retrying in 5s...\n", attempt, maxPreflightRetries)
			time.Sleep(5 * time.Second)
		}
	}
	if !healthy {
		fmt.Fprintln(os.Stderr, "FATAL: Pre-flight /healthz check failed after retries. Is the service running?")
		stats.print()
		os.Exit(1)
	}
	fmt.Println("[pre-flight] Service is healthy")
	fmt.Println()

	// Build a cancellable context for the test duration
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Timer-based cancellation
	if *testTimeMin > 0 {
		go func() {
			timer := time.NewTimer(time.Duration(*testTimeMin) * time.Minute)
			defer timer.Stop()
			select {
			case <-timer.C:
				fmt.Printf("\n[main] Test duration (%d min) reached — shutting down\n", *testTimeMin)
				cancel()
			case <-ctx.Done():
			}
		}()
	}

	// Ctrl-C handling
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		select {
		case sig := <-sigCh:
			fmt.Printf("\n[main] Received %v — shutting down\n", sig)
			cancel()
		case <-ctx.Done():
		}
	}()

	tracker := newRunTracker()
	var wg sync.WaitGroup

	wg.Add(5)
	go thread1_inferenceSubmitter(ctx, &wg, tracker, stats)
	go thread2_statusPoller(ctx, &wg, tracker, stats)
	go thread3_resultDownloader(ctx, &wg, tracker, stats)
	go thread4_miscAPIs(ctx, &wg, stats)
	go thread5_healthWatchdog(ctx, cancel, &wg, stats)

	wg.Wait()

	stats.print()
}
