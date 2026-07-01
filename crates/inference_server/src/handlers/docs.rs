/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use super::*;

/// Swagger UI documentation endpoint.
///
/// The CDN base URL is read from `SWAGGER_CDN_URL` env var at startup.
/// Defaults to `https://unpkg.com/swagger-ui-dist@5`. Set to a local path
/// for air-gapped environments.
pub async fn get_docs(State(state): State<Arc<AppState>>) -> Html<String> {
    let cdn = state
        .config
        .swagger_cdn_url
        .as_deref()
        .unwrap_or("https://unpkg.com/swagger-ui-dist@5");
    Html(format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PhysicsNeMo Serve Inference API - Documentation</title>
    <link rel="stylesheet" type="text/css" href="{cdn}/swagger-ui.css">
    <style>
        html {{ box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }}
        *, *:before, *:after {{ box-sizing: inherit; }}
        body {{ margin: 0; background: #fafafa; }}
        .topbar {{ display: none; }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="{cdn}/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {{
            SwaggerUIBundle({{
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                layout: "BaseLayout"
            }});
        }};
    </script>
</body>
</html>"#
    ))
}

/// Get OpenAPI specification.
pub async fn get_openapi(State(state): State<Arc<AppState>>) -> Json<Value> {
    Json(state.openapi.read().await.clone())
}
