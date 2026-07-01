/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package main

import (
	"context"
	"crypto/tls"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"
)

// ---------------------------------------------------------------------------
// Service abstraction — Rust (PhysicsNeMo Serve) vs Python (Earth2Studio)
// ---------------------------------------------------------------------------

type serviceKind int

const (
	serviceRust   serviceKind = iota
	servicePython serviceKind = iota
)

func (s serviceKind) String() string {
	if s == servicePython {
		return "python"
	}
	return "rust"
}

type serviceConfig struct {
	Kind        serviceKind
	URL         string
	Token       string
	WorkflowID  string
	InputJSON   string
	BasePayload map[string]interface{}
}

const (
	pythonWorkflowID = "ensemble_workflow"
)

// ---------------------------------------------------------------------------
// CLI flags
// ---------------------------------------------------------------------------

var (
	serviceURLRust      = flag.String("service_url_rust", "", "Endpoint URL for the Rust (PhysicsNeMo Serve) service")
	serviceURLPython    = flag.String("service_url_python", "", "Endpoint URL for the Python (Earth2Studio) service")
	inputJSONRust       = flag.String("input_json_rust", "", "Path to base request JSON for Rust service (auto-selected if empty)")
	inputJSONPython     = flag.String("input_json_python", "data/python_service.json", "Path to base request JSON for Python service")
	epToken             = flag.String("ep_token", "", "Bearer token for both services (required)")
	expt                = flag.String("expt", "all", "Experiment to run: 1, 2, 3, or all")
	pollIntervalSec     = flag.Int("poll_interval_sec", 30, "Seconds between status polls during a run")
	runTimeoutMin       = flag.Int("run_timeout_min", 180, "Max minutes to wait for a single inference run")
	outputDir           = flag.String("output_dir", "runs", "Base directory for result output")
	numRetries          = flag.Int("num_retries", 5, "Number of retries for HTTP requests on transient failures (TLS timeouts, etc.)")
)

// sourceDir returns the tests/e2e directory (two levels up from bin/).
func sourceDir() string {
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	binDir := filepath.Dir(exe)
	repoRoot := filepath.Dir(binDir)
	return filepath.Join(repoRoot, "tests", "e2e")
}

// resolvePath resolves a path relative to the source (tests/e2e) directory if not absolute.
func resolvePath(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(sourceDir(), p)
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------

func newHTTPClient() *http.Client {
	return &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig:     &tls.Config{MinVersion: tls.VersionTLS12},
			TLSHandshakeTimeout: 30 * time.Second,
			DialContext: (&net.Dialer{
				Timeout:   30 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
			MaxIdleConns:        20,
			MaxIdleConnsPerHost: 10,
			IdleConnTimeout:     90 * time.Second,
		},
	}
}

func doRequest(ctx context.Context, client *http.Client, method, reqURL, token string, body []byte) (map[string]interface{}, int, error) {
	var lastErr error
	for attempt := 0; attempt <= *numRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(attempt) * 3 * time.Second
			fmt.Printf("    [retry %d/%d after %v] %s %s\n", attempt, *numRetries, backoff, method, reqURL)
			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, 0, ctx.Err()
			}
		}

		var bodyReader io.Reader
		if body != nil {
			bodyReader = strings.NewReader(string(body))
		}
		req, err := http.NewRequestWithContext(ctx, method, reqURL, bodyReader)
		if err != nil {
			return nil, 0, err
		}
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		defer resp.Body.Close()
		data, _ := io.ReadAll(resp.Body)

		var result map[string]interface{}
		_ = json.Unmarshal(data, &result)
		return result, resp.StatusCode, nil
	}
	return nil, 0, fmt.Errorf("all %d retries exhausted: %w", *numRetries, lastErr)
}

// ---------------------------------------------------------------------------
// Inference: submit, poll status, wait for completion
// ---------------------------------------------------------------------------

// submitInfer handles the API differences between Rust and Python services.
//
// Rust:   POST /v1/infer/{workflow}/run   body = payload → response.run_id
// Python: POST /v1/infer/{workflow_name}  body = payload → response.execution_id
func submitInfer(ctx context.Context, client *http.Client, svc *serviceConfig, payload []byte) (string, error) {
	var apiURL string

	switch svc.Kind {
	case serviceRust:
		apiURL = fmt.Sprintf("%s/v1/infer/%s/run", svc.URL, svc.WorkflowID)
	case servicePython:
		apiURL = fmt.Sprintf("%s/v1/infer/%s", svc.URL, svc.WorkflowID)
	}

	result, code, err := doRequest(ctx, client, "POST", apiURL, svc.Token, payload)
	if err != nil {
		return "", fmt.Errorf("[%s] submit request failed: %w", svc.Kind, err)
	}
	if code < 200 || code >= 300 {
		return "", fmt.Errorf("[%s] submit returned HTTP %d: %v", svc.Kind, code, result)
	}

	var runID string
	switch svc.Kind {
	case serviceRust:
		runID, _ = result["run_id"].(string)
	case servicePython:
		runID, _ = result["execution_id"].(string)
	}
	if runID == "" {
		return "", fmt.Errorf("[%s] no run/execution ID in response: %v", svc.Kind, result)
	}
	return runID, nil
}

type runStatusInfo struct {
	Operation string
	Stage     string
	Status    string
}

func queryStatus(ctx context.Context, client *http.Client, svc *serviceConfig, runID string) (runStatusInfo, error) {
	apiURL := fmt.Sprintf("%s/v1/infer/%s/%s/status", svc.URL, svc.WorkflowID, runID)
	result, code, err := doRequest(ctx, client, "GET", apiURL, svc.Token, nil)
	if err != nil {
		return runStatusInfo{}, err
	}
	if code < 200 || code >= 300 {
		return runStatusInfo{}, fmt.Errorf("[%s] status query returned HTTP %d", svc.Kind, code)
	}

	status, _ := result["status"].(string)
	if status == "" {
		if exec, ok := result["execution"].(map[string]interface{}); ok {
			status, _ = exec["status"].(string)
		}
	}
	if status == "" {
		status = "unknown"
	}

	operation, _ := result["operation"].(string)
	stage, _ := result["stage"].(string)

	return runStatusInfo{
		Operation: operation,
		Stage:     stage,
		Status:    strings.ToLower(status),
	}, nil
}

type stageTiming struct {
	Stage    string
	Entered  time.Time
	Duration time.Duration
}

func isTerminal(status string) bool {
	switch status {
	case "completed", "succeeded", "failed", "error", "aborted", "cancelled", "pending_results":
		return true
	}
	return false
}

func isSuccess(status string) bool {
	switch status {
	case "completed", "succeeded", "pending_results":
		return true
	}
	return false
}

// submitAndWait submits an inference request and polls until completion.
// When transitionsOnly is true, only stage transitions (>>) are printed and
// the per-service stage breakdown table is suppressed (caller prints it).
func submitAndWait(ctx context.Context, client *http.Client, svc *serviceConfig, payload []byte, pollInterval, timeout time.Duration, transitionsOnly bool) (time.Duration, string, []stageTiming, error) {
	runID, err := submitInfer(ctx, client, svc, payload)
	if err != nil {
		return 0, "submit_failed", nil, err
	}
	fmt.Printf("    [%s] Submitted id=%s, polling every %v (timeout %v)\n", svc.Kind, runID, pollInterval, timeout)
	start := time.Now()
	deadline := start.Add(timeout)

	var timings []stageTiming
	currentStage := ""

	for {
		select {
		case <-ctx.Done():
			finalizeStageTimings(&timings, currentStage)
			if !transitionsOnly {
				printStageTimings(svc.Kind, runID, timings)
			}
			return time.Since(start), "cancelled", timings, ctx.Err()
		case <-time.After(pollInterval):
		}
		if time.Now().After(deadline) {
			finalizeStageTimings(&timings, currentStage)
			if !transitionsOnly {
				printStageTimings(svc.Kind, runID, timings)
			}
			return time.Since(start), "timeout", timings, fmt.Errorf("run %s timed out after %v", runID, timeout)
		}
		info, err := queryStatus(ctx, client, svc, runID)
		if err != nil {
			if !transitionsOnly {
				fmt.Printf("    [%s][%s] status poll error: %v\n", svc.Kind, runID[:min(8, len(runID))], err)
			}
			continue
		}
		elapsed := time.Since(start)
		idShort := runID[:min(8, len(runID))]

		// Track stage transitions
		stageLabel := info.Stage
		if stageLabel == "" {
			stageLabel = info.Status
		}
		if stageLabel != currentStage {
			now := time.Now()
			finalizeStageTimings(&timings, currentStage)
			timings = append(timings, stageTiming{Stage: stageLabel, Entered: now})
			currentStage = stageLabel
			fmt.Printf("    [%s][%s] >> stage=%-12s  operation=%-5s  status=%-10s  elapsed=%s\n",
				svc.Kind, idShort, stageLabel, info.Operation, info.Status, elapsed.Round(time.Second))
		} else if !transitionsOnly {
			fmt.Printf("    [%s][%s]    stage=%-12s  operation=%-5s  status=%-10s  elapsed=%s\n",
				svc.Kind, idShort, stageLabel, info.Operation, info.Status, elapsed.Round(time.Second))
		}

		if isTerminal(info.Status) {
			finalizeStageTimings(&timings, currentStage)
			if !transitionsOnly {
				printStageTimings(svc.Kind, runID, timings)
			}
			return elapsed, info.Status, timings, nil
		}
	}
}

func finalizeStageTimings(timings *[]stageTiming, currentStage string) {
	if len(*timings) == 0 || currentStage == "" {
		return
	}
	last := &(*timings)[len(*timings)-1]
	if last.Duration == 0 {
		last.Duration = time.Since(last.Entered)
	}
}

func printStageTimings(kind serviceKind, runID string, timings []stageTiming) {
	if len(timings) == 0 {
		return
	}
	idShort := runID[:min(8, len(runID))]
	fmt.Printf("\n    [%s][%s] Stage Breakdown:\n", kind, idShort)
	fmt.Printf("    %-20s %s\n", "STAGE", "WALL-CLOCK DURATION")
	fmt.Printf("    %-20s %s\n", "-----", "-------------------")
	for _, st := range timings {
		fmt.Printf("    %-20s %s\n", st.Stage, st.Duration.Round(time.Second))
	}
	fmt.Println()
}

func writeStageTimingsCSV(runDir, filename string, timings []stageTiming) {
	if len(timings) == 0 {
		return
	}
	headers := []string{"stage", "duration_sec"}
	var rows [][]string
	for _, st := range timings {
		rows = append(rows, []string{st.Stage, ff(st.Duration.Seconds(), 1)})
	}
	csvPath := filepath.Join(runDir, filename)
	if err := writeCSV(csvPath, headers, rows); err != nil {
		fmt.Printf("  ERROR writing stage timings CSV: %v\n", err)
	} else {
		fmt.Printf("    Stage timings CSV: %s\n", csvPath)
	}
}

// ---------------------------------------------------------------------------
// Prometheus range query helpers
// ---------------------------------------------------------------------------

func promRangeAvg(ctx context.Context, client *http.Client, serviceURL, token, query string, start, end time.Time) (float64, error) {
	return promRangeAgg(ctx, client, serviceURL, token, query, start, end, false)
}

func promRangeMax(ctx context.Context, client *http.Client, serviceURL, token, query string, start, end time.Time) (float64, error) {
	return promRangeAgg(ctx, client, serviceURL, token, query, start, end, true)
}

func promRangeAgg(ctx context.Context, client *http.Client, serviceURL, token, query string, start, end time.Time, useMax bool) (float64, error) {
	params := url.Values{}
	params.Set("query", query)
	params.Set("start", strconv.FormatFloat(float64(start.Unix()), 'f', 0, 64))
	params.Set("end", strconv.FormatFloat(float64(end.Unix()), 'f', 0, 64))
	params.Set("step", "15s")

	apiURL := fmt.Sprintf("%s/prometheus/api/v1/query_range?%s", serviceURL, params.Encode())
	result, code, err := doRequest(ctx, client, "GET", apiURL, token, nil)
	if err != nil {
		return 0, fmt.Errorf("prometheus query failed: %w", err)
	}
	if code != 200 {
		return 0, fmt.Errorf("prometheus returned HTTP %d", code)
	}
	data, ok := result["data"].(map[string]interface{})
	if !ok {
		return 0, fmt.Errorf("unexpected prometheus response shape")
	}
	resultArr, ok := data["result"].([]interface{})
	if !ok || len(resultArr) == 0 {
		return 0, fmt.Errorf("empty prometheus result set")
	}

	var sum float64
	var count int
	var maxVal float64

	for _, series := range resultArr {
		seriesMap, ok := series.(map[string]interface{})
		if !ok {
			continue
		}
		values, ok := seriesMap["values"].([]interface{})
		if !ok {
			continue
		}
		for _, point := range values {
			pair, ok := point.([]interface{})
			if !ok || len(pair) < 2 {
				continue
			}
			valStr, ok := pair[1].(string)
			if !ok {
				continue
			}
			val, err := strconv.ParseFloat(valStr, 64)
			if err != nil {
				continue
			}
			sum += val
			count++
			if val > maxVal {
				maxVal = val
			}
		}
	}

	if count == 0 {
		return 0, fmt.Errorf("no valid data points from prometheus")
	}
	if useMax {
		return maxVal, nil
	}
	return sum / float64(count), nil
}

// ---------------------------------------------------------------------------
// JSON payload helpers
// ---------------------------------------------------------------------------

func loadJSON(path string) (map[string]interface{}, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var obj map[string]interface{}
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, err
	}
	return obj, nil
}

func cloneJSON(obj map[string]interface{}) map[string]interface{} {
	out := make(map[string]interface{}, len(obj))
	for k, v := range obj {
		out[k] = v
	}
	return out
}

func marshalPayload(obj map[string]interface{}) []byte {
	data, _ := json.Marshal(obj)
	return data
}

func jsonFloat(obj map[string]interface{}, key string) int {
	if v, ok := obj[key].(float64); ok {
		return int(v)
	}
	// Check inside "parameters" for wrapped payloads (e.g. Python service JSON).
	if params, ok := obj["parameters"].(map[string]interface{}); ok {
		if v, ok := params[key].(float64); ok {
			return int(v)
		}
	}
	return 0
}

// ---------------------------------------------------------------------------
// CSV / output helpers
// ---------------------------------------------------------------------------

func writeCSV(path string, headers []string, rows [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write(headers)
	for _, row := range rows {
		_ = w.Write(row)
	}
	w.Flush()
	return w.Error()
}

func printTable(headers []string, rows [][]string) {
	tw := tabwriter.NewWriter(os.Stdout, 2, 4, 2, ' ', 0)
	fmt.Fprintln(tw, strings.Join(headers, "\t"))
	fmt.Fprintln(tw, strings.Repeat("--------\t", len(headers)))
	for _, row := range rows {
		fmt.Fprintln(tw, strings.Join(row, "\t"))
	}
	tw.Flush()
}

func ff(v float64, prec int) string {
	return strconv.FormatFloat(v, 'f', prec, 64)
}

// ---------------------------------------------------------------------------
// Experiment 1: Wall-Clock Speedup (Baseline Comparison)
// ---------------------------------------------------------------------------

type expt1Result struct {
	Service      string
	GPUs         int
	Nensemble    int
	BatchSize    int
	MaxInFlight  string
	WallClockSec float64
	Status       string
	Stages       []stageTiming
}

func runExperiment1(ctx context.Context, client *http.Client, rustSvc, pythonSvc *serviceConfig, runDir string, batchSizes []int) {
	fmt.Println("\n========== Experiment 1: Wall-Clock Speedup ==========")

	pollInterval := time.Duration(*pollIntervalSec) * time.Second
	timeout := time.Duration(*runTimeoutMin) * time.Minute
	var results []expt1Result

	for _, bs := range batchSizes {
		if ctx.Err() != nil {
			break
		}
		fmt.Printf("\n  --- batch_size=%d ---\n", bs)

		havePython := pythonSvc != nil
		haveRust := rustSvc != nil
		parallel := havePython && haveRust

		var mu sync.Mutex
		var wg sync.WaitGroup

		var pyResult, rustResult expt1Result

		if havePython {
			wg.Add(1)
			go func(batchSize int) {
				defer wg.Done()
			pyPayload := cloneJSON(pythonSvc.BasePayload)
			if params, ok := pyPayload["parameters"].(map[string]interface{}); ok {
				params = cloneJSON(params)
				params["batch_size"] = batchSize
				pyPayload["parameters"] = params
			}
				payload := marshalPayload(pyPayload)
				nensemble := jsonFloat(pythonSvc.BasePayload, "nensemble")
				fmt.Printf("    [python] nensemble=%d batch_size=%d gpus=1\n", nensemble, batchSize)

				elapsed, status, timings, err := submitAndWait(ctx, client, pythonSvc, payload, pollInterval, timeout, parallel)
				if err != nil {
					fmt.Printf("    [python] ERROR: %v\n", err)
				}
				if len(timings) > 0 {
					writeStageTimingsCSV(runDir, fmt.Sprintf("expt1_stages_python_bs%d.csv", batchSize), timings)
				}
				mu.Lock()
				pyResult = expt1Result{
					Service: "python", GPUs: 1, Nensemble: nensemble,
					BatchSize: batchSize, MaxInFlight: "N/A",
					WallClockSec: elapsed.Seconds(), Status: status,
					Stages: timings,
				}
				mu.Unlock()
			}(bs)
		} else {
			pyResult = expt1Result{
				Service: "python", GPUs: 1, BatchSize: bs, MaxInFlight: "N/A", Status: "skipped",
			}
			fmt.Println("    [python] SKIPPED (--service_url_python not set)")
		}

		if haveRust {
			wg.Add(1)
			go func(batchSize int) {
				defer wg.Done()
				rustPayload := cloneJSON(rustSvc.BasePayload)
				rustPayload["max_in_flight"] = 16
				rustPayload["batch_size"] = batchSize
				payload := marshalPayload(rustPayload)
				nensemble := jsonFloat(rustSvc.BasePayload, "nensemble")
				fmt.Printf("    [rust]   nensemble=%d batch_size=%d max_in_flight=16 gpus=8\n", nensemble, batchSize)

				elapsed, status, timings, err := submitAndWait(ctx, client, rustSvc, payload, pollInterval, timeout, parallel)
				if err != nil {
					fmt.Printf("    [rust]   ERROR: %v\n", err)
				}
				if len(timings) > 0 {
					writeStageTimingsCSV(runDir, fmt.Sprintf("expt1_stages_rust_bs%d.csv", batchSize), timings)
				}
				mu.Lock()
				rustResult = expt1Result{
					Service: "rust", GPUs: 8, Nensemble: nensemble,
					BatchSize: batchSize, MaxInFlight: "16",
					WallClockSec: elapsed.Seconds(), Status: status,
					Stages: timings,
				}
				mu.Unlock()
			}(bs)
		} else {
			rustResult = expt1Result{
				Service: "rust", GPUs: 8, BatchSize: bs, MaxInFlight: "16", Status: "skipped",
			}
			fmt.Println("    [rust]   SKIPPED (--service_url_rust not set)")
		}

		wg.Wait()

		results = append(results, pyResult, rustResult)

		if parallel {
			printSideBySideStages(bs, pyResult, rustResult)
		}
	}

	// --- Write CSV ---
	headers := []string{"service", "gpus", "nensemble", "batch_size", "max_in_flight", "wall_clock_sec", "status"}
	var rows [][]string
	for _, r := range results {
		rows = append(rows, []string{
			r.Service, strconv.Itoa(r.GPUs), strconv.Itoa(r.Nensemble),
			strconv.Itoa(r.BatchSize), r.MaxInFlight, ff(r.WallClockSec, 1), r.Status,
		})
	}
	csvPath := filepath.Join(runDir, "experiment_1_results.csv")
	if err := writeCSV(csvPath, headers, rows); err != nil {
		fmt.Printf("  ERROR writing CSV: %v\n", err)
	}

	fmt.Println("\n  Experiment 1 Results:")
	printTable(headers, rows)
	fmt.Printf("  CSV written to %s\n", csvPath)
}

func printSideBySideStages(batchSize int, py, rust expt1Result) {
	fmt.Printf("\n  === Stage Comparison (batch_size=%d) ===\n", batchSize)

	stageSet := make(map[string]bool)
	pyMap := make(map[string]time.Duration)
	rustMap := make(map[string]time.Duration)
	var stageOrder []string

	for _, s := range py.Stages {
		if !stageSet[s.Stage] {
			stageSet[s.Stage] = true
			stageOrder = append(stageOrder, s.Stage)
		}
		pyMap[s.Stage] = s.Duration
	}
	for _, s := range rust.Stages {
		if !stageSet[s.Stage] {
			stageSet[s.Stage] = true
			stageOrder = append(stageOrder, s.Stage)
		}
		rustMap[s.Stage] = s.Duration
	}

	w := tabwriter.NewWriter(os.Stdout, 2, 4, 2, ' ', 0)
	fmt.Fprintf(w, "  %-14s\t%15s\t%15s\n", "STAGE", "PYTHON (1 GPU)", "RUST (8 GPU)")
	fmt.Fprintf(w, "  %-14s\t%15s\t%15s\n", "-----", "--------------", "------------")
	for _, stage := range stageOrder {
		pyDur := "-"
		if d, ok := pyMap[stage]; ok {
			pyDur = d.Round(time.Second).String()
		}
		rustDur := "-"
		if d, ok := rustMap[stage]; ok {
			rustDur = d.Round(time.Second).String()
		}
		fmt.Fprintf(w, "  %-14s\t%15s\t%15s\n", stage, pyDur, rustDur)
	}
	fmt.Fprintf(w, "  %-14s\t%15s\t%15s\n", "TOTAL", fmtDuration(py.WallClockSec), fmtDuration(rust.WallClockSec))
	w.Flush()
	fmt.Println()
}

func fmtDuration(secs float64) string {
	return time.Duration(secs * float64(time.Second)).Round(time.Second).String()
}

// ---------------------------------------------------------------------------
// Experiment 2: GPU Scaling Efficiency (Rust only)
// ---------------------------------------------------------------------------

type expt2Result struct {
	GPUCount     int
	MaxInFlight  int
	BatchSize    int
	Nensemble    int
	WallClockSec float64
	Speedup      float64
	Efficiency   float64
	Status       string
}

func runExperiment2(ctx context.Context, client *http.Client, rustSvc *serviceConfig, runDir string) {
	fmt.Println("\n========== Experiment 2: GPU Scaling Efficiency ==========")

	if rustSvc == nil {
		fmt.Println("  SKIPPED (--service_url_rust not set)")
		return
	}

	const nensemble = 128
	const batchSize = 16
	gpuCounts := []int{1, 2, 4, 8}

	pollInterval := time.Duration(*pollIntervalSec) * time.Second
	timeout := time.Duration(*runTimeoutMin) * time.Minute

	var results []expt2Result
	var baselineTime float64

	for i, gpuCount := range gpuCounts {
		if ctx.Err() != nil {
			break
		}
		fmt.Printf("\n  --- Run %d/%d: gpu_count=%d (max_in_flight=%d, batch_size=%d, nensemble=%d) ---\n", i+1, len(gpuCounts), gpuCount, gpuCount, batchSize, nensemble)

		payload := cloneJSON(rustSvc.BasePayload)
		payload["nensemble"] = nensemble
		payload["max_in_flight"] = gpuCount
		payload["batch_size"] = batchSize
		data := marshalPayload(payload)

		elapsed, status, timings, err := submitAndWait(ctx, client, rustSvc, data, pollInterval, timeout, false)
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
		}
		if len(timings) > 0 {
			writeStageTimingsCSV(runDir, fmt.Sprintf("expt2_stages_gpu%d.csv", gpuCount), timings)
		}

		wallClock := elapsed.Seconds()
		if i == 0 {
			if !isSuccess(status) {
				fmt.Printf("  ABORT: 1-GPU baseline run failed (status=%s). Cannot compute speedups.\n", status)
				results = append(results, expt2Result{
					GPUCount: gpuCount, MaxInFlight: gpuCount, BatchSize: batchSize,
					Nensemble: nensemble, WallClockSec: wallClock,
					Speedup: 0, Efficiency: 0, Status: status,
				})
				break
			}
			baselineTime = wallClock
		}

		speedup := 0.0
		efficiency := 0.0
		if baselineTime > 0 && wallClock > 0 {
			speedup = baselineTime / wallClock
			efficiency = speedup / float64(gpuCount)
		}

		results = append(results, expt2Result{
			GPUCount: gpuCount, MaxInFlight: gpuCount, BatchSize: batchSize,
			Nensemble: nensemble, WallClockSec: wallClock,
			Speedup: speedup, Efficiency: efficiency, Status: status,
		})

		if i < len(gpuCounts)-1 {
			fmt.Println("  Cooldown 15s before next run...")
			time.Sleep(15 * time.Second)
		}
	}

	headers := []string{"gpu_count", "max_in_flight", "batch_size", "nensemble", "wall_clock_sec", "speedup", "efficiency", "status"}
	var rows [][]string
	for _, r := range results {
		rows = append(rows, []string{
			strconv.Itoa(r.GPUCount), strconv.Itoa(r.MaxInFlight), strconv.Itoa(r.BatchSize),
			strconv.Itoa(r.Nensemble), ff(r.WallClockSec, 1), ff(r.Speedup, 2), ff(r.Efficiency, 3), r.Status,
		})
	}
	csvPath := filepath.Join(runDir, "experiment_2_results.csv")
	if err := writeCSV(csvPath, headers, rows); err != nil {
		fmt.Printf("  ERROR writing CSV: %v\n", err)
	}

	fmt.Println("\n  Experiment 2 Results:")
	printTable(headers, rows)
	fmt.Printf("  CSV written to %s\n", csvPath)
}

// ---------------------------------------------------------------------------
// Experiment 3: Batch Size Sensitivity (Rust only)
// ---------------------------------------------------------------------------

type expt3Result struct {
	BatchSize    int
	Nensemble    int
	MaxInFlight  int
	WallClockSec float64
	AvgGPUComp   float64
	AvgGPUMemMB  float64
	PeakGPUMemMB float64
	Status       string
	MetricsErr   string
}

func runExperiment3(ctx context.Context, client *http.Client, rustSvc *serviceConfig, runDir string) {
	fmt.Println("\n========== Experiment 3: Batch Size Sensitivity ==========")

	if rustSvc == nil {
		fmt.Println("  SKIPPED (--service_url_rust not set)")
		return
	}

	const nensemble = 512
	const expt3MaxInFlight = 16
	batchSizes := []int{16, 32, 64}

	pollInterval := time.Duration(*pollIntervalSec) * time.Second
	timeout := time.Duration(*runTimeoutMin) * time.Minute

	var results []expt3Result

	for i, bs := range batchSizes {
		if ctx.Err() != nil {
			break
		}
		fmt.Printf("\n  --- Run %d/%d: batch_size=%d ---\n", i+1, len(batchSizes), bs)

		payload := cloneJSON(rustSvc.BasePayload)
		payload["nensemble"] = nensemble
		payload["batch_size"] = bs
		payload["max_in_flight"] = expt3MaxInFlight
		data := marshalPayload(payload)

		runStart := time.Now()
		elapsed, status, timings, err := submitAndWait(ctx, client, rustSvc, data, pollInterval, timeout, false)
		runEnd := time.Now()
		if err != nil {
			fmt.Printf("  ERROR: %v\n", err)
		}
		if len(timings) > 0 {
			writeStageTimingsCSV(runDir, fmt.Sprintf("expt3_stages_bs%d.csv", bs), timings)
		}

		r := expt3Result{
			BatchSize: bs, Nensemble: nensemble, MaxInFlight: expt3MaxInFlight,
			WallClockSec: elapsed.Seconds(), Status: status,
		}

		// Query Prometheus for GPU metrics over the run duration (with 15s buffer).
		promStart := runStart.Add(-15 * time.Second)
		promEnd := runEnd.Add(15 * time.Second)

		if avgComp, err := promRangeAvg(ctx, client, rustSvc.URL, rustSvc.Token,
			"avg(physicsnemo_serve_gpu_compute_utilization_percent)", promStart, promEnd); err != nil {
			r.MetricsErr = err.Error()
			fmt.Printf("    [warn] GPU compute metrics unavailable: %v\n", err)
		} else {
			r.AvgGPUComp = avgComp
		}

		if avgMem, err := promRangeAvg(ctx, client, rustSvc.URL, rustSvc.Token,
			"avg(physicsnemo_serve_gpu_memory_used_bytes)", promStart, promEnd); err != nil {
			if r.MetricsErr == "" {
				r.MetricsErr = err.Error()
			}
		} else {
			r.AvgGPUMemMB = avgMem / (1024 * 1024)
		}

		if peakMem, err := promRangeMax(ctx, client, rustSvc.URL, rustSvc.Token,
			"max(physicsnemo_serve_gpu_memory_used_bytes)", promStart, promEnd); err != nil {
			if r.MetricsErr == "" {
				r.MetricsErr = err.Error()
			}
		} else {
			r.PeakGPUMemMB = peakMem / (1024 * 1024)
		}

		results = append(results, r)

		if i < len(batchSizes)-1 {
			fmt.Println("  Cooldown 15s before next run...")
			time.Sleep(15 * time.Second)
		}
	}

	headers := []string{"batch_size", "nensemble", "max_in_flight", "wall_clock_sec",
		"avg_gpu_compute_pct", "avg_gpu_mem_used_mb", "peak_gpu_mem_used_mb", "status"}
	var rows [][]string
	for _, r := range results {
		rows = append(rows, []string{
			strconv.Itoa(r.BatchSize), strconv.Itoa(r.Nensemble), strconv.Itoa(r.MaxInFlight),
			ff(r.WallClockSec, 1), ff(r.AvgGPUComp, 1), ff(r.AvgGPUMemMB, 0),
			ff(r.PeakGPUMemMB, 0), r.Status,
		})
	}
	csvPath := filepath.Join(runDir, "experiment_3_results.csv")
	if err := writeCSV(csvPath, headers, rows); err != nil {
		fmt.Printf("  ERROR writing CSV: %v\n", err)
	}

	fmt.Println("\n  Experiment 3 Results:")
	printTable(headers, rows)
	fmt.Printf("  CSV written to %s\n", csvPath)
}

// ---------------------------------------------------------------------------
// Summary JSON
// ---------------------------------------------------------------------------

func writeSummary(runDir string, startTime time.Time, rustSvc, pythonSvc *serviceConfig) {
	summary := map[string]interface{}{
		"run_dir":    runDir,
		"experiment": *expt,
		"start_time": startTime.UTC().Format(time.RFC3339),
		"end_time":   time.Now().UTC().Format(time.RFC3339),
	}
	if rustSvc != nil {
		summary["service_url_rust"] = rustSvc.URL
		summary["input_json_rust"] = rustSvc.InputJSON
	}
	if pythonSvc != nil {
		summary["service_url_python"] = pythonSvc.URL
		summary["input_json_python"] = pythonSvc.InputJSON
	}
	data, _ := json.MarshalIndent(summary, "", "  ")
	path := filepath.Join(runDir, "summary.json")
	os.WriteFile(path, data, 0644)
	fmt.Printf("\n  Summary written to %s\n", path)
}

// ---------------------------------------------------------------------------
// Plot script generator (Python + matplotlib)
// ---------------------------------------------------------------------------

func writePlotScript(runDir string) {
	script := `#!/usr/bin/env python3
"""Auto-generated plot script for ensemble performance experiments.

Usage:  python plot_results.py
Reads CSVs from the current directory, writes PNG plots alongside them.
"""
import csv
import os
import sys

def read_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return list(reader.fieldnames or []), rows

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed — skipping plots. pip install matplotlib")
    sys.exit(0)


def plot_experiment_1():
    """Experiment 1: Bar chart comparing Python vs Rust wall-clock time."""
    _, rows = read_csv("experiment_1_results.csv")
    if not rows:
        print("No Experiment 1 data found, skipping.")
        return
    services = []
    times = []
    for r in rows:
        if r["status"] in ("skipped", "not_run", ""):
            continue
        services.append(f"{r['service'].capitalize()}\n({r['gpus']} GPU)")
        times.append(float(r["wall_clock_sec"]))
    if not services:
        print("No completed Experiment 1 runs, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4A90D9" if "Python" in s else "#E07B39" for s in services]
    bars = ax.bar(services, times, color=colors, width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(times) * 0.02,
                f"{t:.0f}s", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Wall-Clock Time (s)")
    ax.set_title("Experiment 1: Wall-Clock Speedup")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_1_plot.png", dpi=150)
    plt.close()
    print("Saved experiment_1_plot.png")


def plot_experiment_2():
    """GPU Scaling: wall-clock, speedup, efficiency vs GPU count."""
    _, rows = read_csv("experiment_2_results.csv")
    if not rows:
        print("No Experiment 2 data found, skipping.")
        return

    gpu_counts = [int(r["gpu_count"]) for r in rows]
    wall_clocks = [float(r["wall_clock_sec"]) for r in rows]
    speedups = [float(r["speedup"]) for r in rows]
    efficiencies = [float(r["efficiency"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(gpu_counts, wall_clocks, "o-", linewidth=2)
    axes[0].set_xlabel("GPU Count (simulated via max_in_flight)")
    axes[0].set_ylabel("Wall-Clock Time (s)")
    axes[0].set_title("Wall-Clock Time vs GPU Count")
    axes[0].set_xticks(gpu_counts)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gpu_counts, speedups, "o-", linewidth=2, label="Actual")
    axes[1].plot(gpu_counts, gpu_counts, "--", color="gray", alpha=0.6, label="Ideal linear")
    axes[1].set_xlabel("GPU Count")
    axes[1].set_ylabel("Speedup")
    axes[1].set_title("Speedup vs GPU Count")
    axes[1].set_xticks(gpu_counts)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(gpu_counts, efficiencies, "o-", linewidth=2)
    axes[2].axhline(y=1.0, linestyle="--", color="gray", alpha=0.6, label="Ideal (1.0)")
    axes[2].set_xlabel("GPU Count")
    axes[2].set_ylabel("Efficiency (speedup / GPUs)")
    axes[2].set_title("Scaling Efficiency")
    axes[2].set_xticks(gpu_counts)
    axes[2].set_ylim(0, 1.2)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_2_plots.png", dpi=150)
    plt.close()
    print("Saved experiment_2_plots.png")


def plot_experiment_3():
    """Batch Size Sensitivity: wall-clock, GPU compute, GPU memory vs batch_size."""
    _, rows = read_csv("experiment_3_results.csv")
    if not rows:
        print("No Experiment 3 data found, skipping.")
        return

    batch_sizes = [int(r["batch_size"]) for r in rows]
    wall_clocks = [float(r["wall_clock_sec"]) for r in rows]
    gpu_compute = [float(r["avg_gpu_compute_pct"]) for r in rows]
    gpu_mem_peak = [float(r["peak_gpu_mem_used_mb"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(batch_sizes, wall_clocks, "o-", linewidth=2)
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("Wall-Clock Time (s)")
    axes[0].set_title("Wall-Clock Time vs Batch Size")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(batch_sizes)
    axes[0].set_xticklabels([str(b) for b in batch_sizes])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(batch_sizes, gpu_compute, "o-", linewidth=2, color="tab:orange")
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("Avg GPU Compute Utilization (%)")
    axes[1].set_title("GPU Compute vs Batch Size")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(batch_sizes)
    axes[1].set_xticklabels([str(b) for b in batch_sizes])
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(batch_sizes, gpu_mem_peak, "o-", linewidth=2, color="tab:red")
    axes[2].set_xlabel("Batch Size")
    axes[2].set_ylabel("Peak GPU Memory Used (MiB)")
    axes[2].set_title("GPU Memory vs Batch Size")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(batch_sizes)
    axes[2].set_xticklabels([str(b) for b in batch_sizes])
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiment_3_plots.png", dpi=150)
    plt.close()
    print("Saved experiment_3_plots.png")


if __name__ == "__main__":
    plot_experiment_1()
    plot_experiment_2()
    plot_experiment_3()
    print("Done.")
`
	path := filepath.Join(runDir, "plot_results.py")
	os.WriteFile(path, []byte(script), 0755)
	fmt.Printf("  Plot script written to %s\n", path)
	fmt.Printf("  Run: cd %s && python plot_results.py\n", runDir)
}

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------

func healthCheck(ctx context.Context, client *http.Client, serviceURL, token, healthPath string) error {
	for attempt := 1; attempt <= 3; attempt++ {
		apiURL := serviceURL + healthPath
		_, code, err := doRequest(ctx, client, "GET", apiURL, token, nil)
		if err == nil && code >= 200 && code < 300 {
			return nil
		}
		if attempt < 3 {
			fmt.Printf("    Health check attempt %d failed (err=%v, code=%d), retrying in 5s...\n", attempt, err, code)
			time.Sleep(5 * time.Second)
		} else {
			if err != nil {
				return fmt.Errorf("health check failed after %d attempts: %w", attempt, err)
			}
			return fmt.Errorf("health check failed after %d attempts: HTTP %d", attempt, code)
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func main() {
	flag.Parse()

	if *serviceURLRust == "" && *serviceURLPython == "" {
		fmt.Fprintln(os.Stderr, "ERROR: at least one of --service_url_rust or --service_url_python is required")
		flag.Usage()
		os.Exit(1)
	}
	if *epToken == "" {
		fmt.Fprintln(os.Stderr, "ERROR: --ep_token is required")
		flag.Usage()
		os.Exit(1)
	}

	validExpts := map[string]bool{"1": true, "2": true, "3": true, "all": true}
	if !validExpts[*expt] {
		fmt.Fprintf(os.Stderr, "ERROR: --expt must be 1, 2, 3, or all (got %q)\n", *expt)
		os.Exit(1)
	}

	rustWorkflowID := "earth2-ensemble-fanout"
	if *inputJSONRust == "" {
		*inputJSONRust = "data/rust_service_earth2.json"
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	client := newHTTPClient()

	// --- Build service configs ---
	var rustSvc, pythonSvc *serviceConfig

	if *serviceURLRust != "" {
		url := strings.TrimRight(*serviceURLRust, "/")
		fmt.Printf("Rust workflow: %s (input: %s)\n", rustWorkflowID, *inputJSONRust)
		fmt.Printf("Checking Rust service health at %s ...\n", url)
		if err := healthCheck(ctx, client, url, *epToken, "/healthz"); err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: Rust service not healthy: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("  Rust service is healthy.")
		rustJSONPath := resolvePath(*inputJSONRust)
		base, err := loadJSON(rustJSONPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: cannot load %s: %v\n", rustJSONPath, err)
			os.Exit(1)
		}
		rustSvc = &serviceConfig{
			Kind: serviceRust, URL: url, Token: *epToken,
			WorkflowID: rustWorkflowID, InputJSON: *inputJSONRust, BasePayload: base,
		}
	}

	if *serviceURLPython != "" {
		url := strings.TrimRight(*serviceURLPython, "/")
		fmt.Printf("Checking Python service health at %s ...\n", url)
		if err := healthCheck(ctx, client, url, *epToken, "/health"); err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: Python service not healthy: %v\n", err)
			os.Exit(1)
		}
		fmt.Println("  Python service is healthy.")
		pythonJSONPath := resolvePath(*inputJSONPython)
		base, err := loadJSON(pythonJSONPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: cannot load %s: %v\n", pythonJSONPath, err)
			os.Exit(1)
		}
		pythonSvc = &serviceConfig{
			Kind: servicePython, URL: url, Token: *epToken,
			WorkflowID: pythonWorkflowID, InputJSON: *inputJSONPython, BasePayload: base,
		}
	}

	// --- Create run directory ---
	salt := fmt.Sprintf("%06x", rand.Intn(0xFFFFFF))
	timestamp := time.Now().Format("20060102_150405")
	runDirName := fmt.Sprintf("run_%s_%s", timestamp, salt)
	runDir := filepath.Join(*outputDir, runDirName)
	if err := os.MkdirAll(runDir, 0755); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: cannot create run directory: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Run directory: %s\n", runDir)

	startTime := time.Now()

	// --- Run experiments ---
	expt1BatchSizes := []int{16, 32, 64}

	switch *expt {
	case "1":
		runExperiment1(ctx, client, rustSvc, pythonSvc, runDir, expt1BatchSizes)
	case "2":
		runExperiment2(ctx, client, rustSvc, runDir)
	case "3":
		runExperiment3(ctx, client, rustSvc, runDir)
	case "all":
		runExperiment1(ctx, client, rustSvc, pythonSvc, runDir, expt1BatchSizes)
		runExperiment2(ctx, client, rustSvc, runDir)
		runExperiment3(ctx, client, rustSvc, runDir)
	}

	writeSummary(runDir, startTime, rustSvc, pythonSvc)
	writePlotScript(runDir)

	totalElapsed := time.Since(startTime)
	fmt.Printf("\n========== All experiments completed in %s ==========\n", totalElapsed.Round(time.Second))

}
