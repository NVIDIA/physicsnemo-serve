/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GFS (Global Forecast System) data source utilities.
//!
//! This module provides:
//! - GFS variable name to GRIB pattern mapping (lexicon)
//! - GFS file path construction utilities

use std::collections::HashMap;

/// Returns the GFS lexicon mapping earth2studio variable names to GRIB patterns.
///
/// The lexicon maps short variable names (e.g., "t2m", "z500") to their
/// corresponding GRIB variable name and level description patterns.
///
/// # Example
///
/// ```ignore
/// let lexicon = gfs_lexicon();
/// assert_eq!(lexicon.get("t2m"), Some(&"TMP::2 m above ground".to_string()));
/// assert_eq!(lexicon.get("z500"), Some(&"HGT::500 mb".to_string()));
/// ```
pub fn gfs_lexicon() -> HashMap<String, String> {
    let mut lexicon = HashMap::new();

    // Temperature
    lexicon.insert("t2m".to_string(), "TMP::2 m above ground".to_string());
    lexicon.insert("t250".to_string(), "TMP::250 mb".to_string());
    lexicon.insert("t500".to_string(), "TMP::500 mb".to_string());
    lexicon.insert("t850".to_string(), "TMP::850 mb".to_string());

    // Geopotential height
    lexicon.insert("z50".to_string(), "HGT::50 mb".to_string());
    lexicon.insert("z250".to_string(), "HGT::250 mb".to_string());
    lexicon.insert("z300".to_string(), "HGT::300 mb".to_string());
    lexicon.insert("z500".to_string(), "HGT::500 mb".to_string());
    lexicon.insert("z700".to_string(), "HGT::700 mb".to_string());
    lexicon.insert("z850".to_string(), "HGT::850 mb".to_string());
    lexicon.insert("z1000".to_string(), "HGT::1000 mb".to_string());

    // U-wind
    lexicon.insert("u10m".to_string(), "UGRD::10 m above ground".to_string());
    lexicon.insert("u100m".to_string(), "UGRD::100 m above ground".to_string());
    lexicon.insert("u250".to_string(), "UGRD::250 mb".to_string());
    lexicon.insert("u500".to_string(), "UGRD::500 mb".to_string());
    lexicon.insert("u850".to_string(), "UGRD::850 mb".to_string());
    lexicon.insert("u1000".to_string(), "UGRD::1000 mb".to_string());

    // V-wind
    lexicon.insert("v10m".to_string(), "VGRD::10 m above ground".to_string());
    lexicon.insert("v100m".to_string(), "VGRD::100 m above ground".to_string());
    lexicon.insert("v250".to_string(), "VGRD::250 mb".to_string());
    lexicon.insert("v500".to_string(), "VGRD::500 mb".to_string());
    lexicon.insert("v850".to_string(), "VGRD::850 mb".to_string());
    lexicon.insert("v1000".to_string(), "VGRD::1000 mb".to_string());

    // Relative humidity
    lexicon.insert("r500".to_string(), "RH::500 mb".to_string());
    lexicon.insert("r850".to_string(), "RH::850 mb".to_string());

    // Other
    lexicon.insert("sp".to_string(), "PRES::surface".to_string());
    lexicon.insert("msl".to_string(), "PRMSL::mean sea level".to_string());
    lexicon.insert("tcwv".to_string(), "PWAT::entire atmosphere".to_string());

    lexicon
}

/// Constructs a GFS file path for the given time and lead hour.
///
/// # Arguments
///
/// * `time` - The forecast initialization time
/// * `lead_hour` - The forecast lead time in hours
/// * `uri_prefix` - The S3 bucket prefix (e.g., "noaa-gfs-bdp-pds")
/// * `suffix` - File suffix (e.g., "" for GRIB, ".idx" for index)
///
/// # Example
///
/// ```ignore
/// let time = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
/// let path = construct_gfs_path(&time, 6, "noaa-gfs-bdp-pds", ".idx");
/// assert!(path.contains("gfs.20240115"));
/// assert!(path.contains("f006.idx"));
/// ```
pub fn construct_gfs_path(
    time: &chrono::DateTime<chrono::Utc>,
    lead_hour: i32,
    uri_prefix: &str,
    suffix: &str,
) -> String {
    use chrono::{Datelike, Timelike};

    format!(
        "{}/gfs.{}{:02}{:02}/{:02}/atmos/gfs.t{:02}z.pgrb2.0p25.f{:03}{}",
        uri_prefix,
        time.year(),
        time.month(),
        time.day(),
        time.hour(),
        time.hour(),
        lead_hour.unsigned_abs(),
        suffix
    )
}

/// Parses a GFS .idx file content to extract byte ranges for specified variables.
///
/// # Arguments
///
/// * `idx_content` - The raw content of the .idx file
/// * `variables` - List of variable names to extract
/// * `lexicon` - The GFS lexicon for variable name mapping
///
/// # Returns
///
/// A HashMap mapping variable names to (byte_offset, byte_length) tuples.
pub fn parse_idx_for_variables(
    idx_content: &str,
    variables: &[String],
    lexicon: &HashMap<String, String>,
) -> HashMap<String, (u64, u64)> {
    let mut byte_ranges = HashMap::new();
    let lines: Vec<&str> = idx_content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() < 4 {
            continue;
        }

        let byte_offset = match parts[1].parse::<u64>() {
            Ok(offset) => offset,
            Err(_) => continue,
        };

        let byte_length = if i + 1 < lines.len() {
            let next_parts: Vec<&str> = lines[i + 1].split(':').collect();
            if next_parts.len() >= 2 {
                match next_parts[1].parse::<u64>() {
                    Ok(next_offset) => next_offset - byte_offset,
                    Err(_) => continue,
                }
            } else {
                continue;
            }
        } else {
            continue;
        };

        for var in variables {
            if byte_ranges.contains_key(var) {
                continue;
            }

            if let Some(grib_pattern) = lexicon.get(var)
                && let Some((grib_name, level_desc)) = grib_pattern.split_once("::")
                && line.contains(grib_name)
                && line.contains(level_desc)
            {
                byte_ranges.insert(var.clone(), (byte_offset, byte_length));
            }
        }
    }

    byte_ranges
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};

    #[test]
    fn lexicon_maps_t2m_to_grib_pattern() {
        let lexicon = gfs_lexicon();
        assert_eq!(
            lexicon.get("t2m"),
            Some(&"TMP::2 m above ground".to_string())
        );
    }

    #[test]
    fn lexicon_maps_z500_to_grib_pattern() {
        let lexicon = gfs_lexicon();
        assert_eq!(lexicon.get("z500"), Some(&"HGT::500 mb".to_string()));
    }

    #[test]
    fn lexicon_contains_all_temperature_levels() {
        let lexicon = gfs_lexicon();
        for var in ["t2m", "t250", "t500", "t850"] {
            assert!(lexicon.contains_key(var), "missing {var}");
        }
    }

    #[test]
    fn lexicon_contains_all_geopotential_levels() {
        let lexicon = gfs_lexicon();
        for var in ["z50", "z250", "z300", "z500", "z700", "z850", "z1000"] {
            assert!(lexicon.contains_key(var), "missing {var}");
        }
    }

    #[test]
    fn lexicon_contains_all_wind_variables() {
        let lexicon = gfs_lexicon();
        for var in [
            "u10m", "u100m", "u250", "u500", "u850", "u1000", "v10m", "v100m", "v250", "v500",
            "v850", "v1000",
        ] {
            assert!(lexicon.contains_key(var), "missing {var}");
        }
    }

    #[test]
    fn lexicon_contains_surface_and_humidity_variables() {
        let lexicon = gfs_lexicon();
        for var in ["sp", "msl", "tcwv", "r500", "r850"] {
            assert!(lexicon.contains_key(var), "missing {var}");
        }
    }

    #[test]
    fn lexicon_has_28_total_entries() {
        let lexicon = gfs_lexicon();
        assert_eq!(lexicon.len(), 28);
    }

    #[test]
    fn construct_path_includes_date_hour_and_lead() {
        let time = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let path = construct_gfs_path(&time, 6, "noaa-gfs-bdp-pds", "");

        assert!(path.starts_with("noaa-gfs-bdp-pds/"));
        assert!(path.contains("gfs.20240115"));
        assert!(path.contains("/12/atmos/"));
        assert!(path.contains("gfs.t12z.pgrb2.0p25.f006"));
    }

    #[test]
    fn construct_path_with_idx_suffix() {
        let time = Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap();
        let path = construct_gfs_path(&time, 6, "noaa-gfs-bdp-pds", ".idx");

        assert!(path.ends_with(".idx"));
    }

    #[test]
    fn construct_path_zero_lead_hour() {
        let time = Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap();
        let path = construct_gfs_path(&time, 0, "noaa-gfs-bdp-pds", "");

        assert!(path.contains("f000"));
        assert!(path.contains("gfs.t00z"));
    }

    #[test]
    fn construct_path_large_lead_hour() {
        let time = Utc.with_ymd_and_hms(2024, 1, 15, 6, 0, 0).unwrap();
        let path = construct_gfs_path(&time, 120, "noaa-gfs-bdp-pds", "");

        assert!(path.contains("f120"));
    }

    #[test]
    fn parse_idx_extracts_byte_ranges_for_known_variables() {
        let idx_content = r#"1:0:d=2024011512:TMP:2 m above ground:anl:
2:100000:d=2024011512:TMP:250 mb:anl:
3:200000:d=2024011512:HGT:500 mb:anl:
4:300000:d=2024011512:END"#;

        let variables = vec!["t2m".to_string(), "z500".to_string()];
        let lexicon = gfs_lexicon();
        let ranges = parse_idx_for_variables(idx_content, &variables, &lexicon);

        assert_eq!(ranges.get("t2m"), Some(&(0, 100000)));
        assert_eq!(ranges.get("z500"), Some(&(200000, 100000)));
    }

    #[test]
    fn parse_idx_returns_empty_for_missing_variable() {
        let idx_content = r#"1:0:d=2024011512:TMP:2 m above ground:anl:
2:100000:d=2024011512:END"#;

        let variables = vec!["z500".to_string()];
        let lexicon = gfs_lexicon();
        let ranges = parse_idx_for_variables(idx_content, &variables, &lexicon);

        assert!(ranges.is_empty());
    }

    #[test]
    fn parse_idx_returns_empty_for_empty_content() {
        let idx_content = "";
        let variables = vec!["t2m".to_string()];
        let lexicon = gfs_lexicon();
        let ranges = parse_idx_for_variables(idx_content, &variables, &lexicon);

        assert!(ranges.is_empty());
    }
}
