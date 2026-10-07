"""DoMINO surface prediction from prepared local and surface neighborhoods.

Spatial queries, grid projection and neighbor gathering stay outside this graph.
The learned local encoders, positional encoder and solution head remain inside.
"""


def create_model(config, assets):
    import torch
    from physicsnemo.models.domino import DoMINO

    parameters = config["model_parameters"]
    if (
        config.get("input_features") != 3
        or config.get("output_features_vol") is not None
        or config.get("output_features_surf") != 4
        or parameters.get("combine_volume_surface") is not False
        or parameters.get("encode_parameters") is not False
        or parameters.get("geometry_encoding_type") != "both"
        or parameters.get("num_neighbors_surface") != 7
        or parameters.get("use_surface_normals") is not True
        or parameters.get("use_surface_area") is not True
        or len(parameters["geometry_local"]["surface_neighbors_in_radius"]) != 2
    ):
        raise ValueError(
            "This example requires the single-stream DoMINO surface checkpoint."
        )

    class SurfaceCore(DoMINO):
        # Preserve upstream state names for strict loading of imported weights.
        def forward(
            self,
            local_neighbors_0,
            local_neighbors_1,
            centers,
            relative_positions,
            neighbor_centers,
            normals,
            neighbor_normals,
            areas,
            neighbor_areas,
        ):
            encoders = self.surface_local_geo_encodings.local_geo_encodings
            local = torch.cat(
                (
                    encoders[0].local_point_conv(local_neighbors_0),
                    encoders[1].local_point_conv(local_neighbors_1),
                ),
                dim=-1,
            )
            position = self.fc_p_surf(relative_positions)
            return self.solution_calculator_surf(
                centers,
                local,
                position,
                neighbor_centers,
                normals,
                neighbor_normals,
                areas,
                neighbor_areas,
                None,
                None,
            )

    return SurfaceCore(**config)


def create_cases(config, assets):
    """Three deterministic model-space cases, with positive surface cell areas."""
    import torch

    parameters = config["model_parameters"]
    channels = 1 + len(parameters["geometry_rep"]["geo_conv"]["surface_radii"])
    scales = parameters["geometry_local"]["surface_neighbors_in_radius"]
    neighbors = parameters["num_neighbors_surface"] - 1
    generator = torch.Generator().manual_seed(42)

    def random(*shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1

    cases = []
    for _ in range(3):
        centers = random(1, 32, 3)
        normals = torch.nn.functional.normalize(random(1, 32, 3), dim=-1)
        neighbor_centers = centers.unsqueeze(2) + random(1, 32, neighbors, 3)
        neighbor_normals = torch.nn.functional.normalize(
            random(1, 32, neighbors, 3), dim=-1
        )
        cases.append(
            (
                random(1, 32, scales[0] * channels),
                random(1, 32, scales[1] * channels),
                centers,
                centers - centers.mean(dim=1, keepdim=True),
                neighbor_centers,
                normals,
                neighbor_normals,
                random(1, 32, 1).abs() + 0.01,
                random(1, 32, neighbors, 1).abs() + 0.01,
            )
        )
    return cases
