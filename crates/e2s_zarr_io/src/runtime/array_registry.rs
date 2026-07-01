/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Array registration contract management for `add_array()` and `write()`.

use std::collections::HashSet;
use std::path::{Component, Path};
use std::sync::Mutex;

use crate::core::contracts::ArrayRegistry;
use crate::core::errors::SyncWriteError;
use crate::core::types::{ArrayRegistration, CoordMap};

struct RegisteredState {
    registration: ArrayRegistration,
    array_id_by_name: std::collections::HashMap<String, u32>,
}

/// In-memory `ArrayRegistry` for lifecycle contract enforcement.
///
/// - Registration is single-shot (`add_array()` once).
/// - Write-time array names must be a subset of registered names.
pub struct InMemoryArrayRegistry {
    registered: Mutex<Option<RegisteredState>>,
}

impl Default for InMemoryArrayRegistry {
    fn default() -> Self {
        Self {
            registered: Mutex::new(None),
        }
    }
}

impl InMemoryArrayRegistry {
    /// Create a new empty array registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn validate_safe_dataset_component(name: &str, field: &str) -> Result<(), SyncWriteError> {
    let path = Path::new(name);
    let mut components = path.components();
    let Some(first) = components.next() else {
        return Err(SyncWriteError::Validation {
            message: format!("add_array() {field} must be a single safe path component"),
        });
    };
    let has_extra_components = components.next().is_some();
    if path.is_absolute()
        || has_extra_components
        || !matches!(first, Component::Normal(_))
        || name.contains('\\')
    {
        return Err(SyncWriteError::Validation {
            message: format!("add_array() {field} '{name}' must be a single safe path component"),
        });
    }
    Ok(())
}

impl ArrayRegistry for InMemoryArrayRegistry {
    fn register(&self, req: ArrayRegistration) -> Result<(), SyncWriteError> {
        if req.array_names.is_empty() {
            return Err(SyncWriteError::Validation {
                message: "add_array() requires at least one array name".to_string(),
            });
        }
        if req.array_names.iter().any(|name| name.trim().is_empty()) {
            return Err(SyncWriteError::Validation {
                message: "add_array() requires non-empty array names".to_string(),
            });
        }

        let unique_names: HashSet<&str> = req.array_names.iter().map(String::as_str).collect();
        if unique_names.len() != req.array_names.len() {
            return Err(SyncWriteError::Validation {
                message: "add_array() array_names must be unique".to_string(),
            });
        }
        for array_name in &req.array_names {
            validate_safe_dataset_component(array_name, "array name")?;
        }
        for coord_name in req.coords.keys() {
            validate_safe_dataset_component(coord_name, "coord name")?;
        }
        if let Some((coord_name, _)) = req.coords.iter().find(|(_, values)| values.is_empty()) {
            return Err(SyncWriteError::Validation {
                message: format!("add_array() coord '{coord_name}' must have at least one value"),
            });
        }

        let mut registered =
            self.registered
                .lock()
                .map_err(|_| SyncWriteError::ContractViolation {
                    message: "registration lock poisoned".to_string(),
                })?;
        if registered.is_some() {
            return Err(SyncWriteError::ContractViolation {
                message: "add_array() can only be called once in v1".to_string(),
            });
        }

        let mut array_id_by_name = std::collections::HashMap::with_capacity(req.array_names.len());
        for (idx, name) in req.array_names.iter().enumerate() {
            let array_id = u32::try_from(idx).map_err(|_| SyncWriteError::Validation {
                message: "array registration index overflowed u32".to_string(),
            })?;
            array_id_by_name.insert(name.clone(), array_id);
        }

        *registered = Some(RegisteredState {
            registration: req,
            array_id_by_name,
        });
        Ok(())
    }

    fn validate_write_array_names(&self, array_names: &[String]) -> Result<(), SyncWriteError> {
        let registered = self
            .registered
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "registration lock poisoned".to_string(),
            })?;
        let Some(registered) = &*registered else {
            return Err(SyncWriteError::ContractViolation {
                message: "write() called before add_array()".to_string(),
            });
        };

        let known: HashSet<&str> = registered
            .registration
            .array_names
            .iter()
            .map(String::as_str)
            .collect();
        if let Some(unknown) = array_names
            .iter()
            .find(|name| !known.contains(name.as_str()))
        {
            return Err(SyncWriteError::ContractViolation {
                message: format!("write() references unknown array name: {unknown}"),
            });
        }

        Ok(())
    }

    fn resolve_array_ids(&self, array_names: &[String]) -> Result<Vec<u32>, SyncWriteError> {
        let registered = self
            .registered
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "registration lock poisoned".to_string(),
            })?;
        let Some(registered) = &*registered else {
            return Err(SyncWriteError::ContractViolation {
                message: "write() called before add_array()".to_string(),
            });
        };

        let mut ids = Vec::with_capacity(array_names.len());
        for name in array_names {
            let array_id = registered
                .array_id_by_name
                .get(name)
                .copied()
                .ok_or_else(|| SyncWriteError::ContractViolation {
                    message: format!("write() references unknown array name: {name}"),
                })?;
            ids.push(array_id);
        }

        Ok(ids)
    }

    fn registered_coords(&self) -> Result<CoordMap, SyncWriteError> {
        let registered = self
            .registered
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "registration lock poisoned".to_string(),
            })?;
        let Some(registered) = &*registered else {
            return Err(SyncWriteError::ContractViolation {
                message: "write() called before add_array()".to_string(),
            });
        };

        Ok(registered.registration.coords.clone())
    }

    fn registration_snapshot(&self) -> Result<Option<ArrayRegistration>, SyncWriteError> {
        let registered = self
            .registered
            .lock()
            .map_err(|_| SyncWriteError::ContractViolation {
                message: "registration lock poisoned".to_string(),
            })?;
        Ok(registered.as_ref().map(|state| state.registration.clone()))
    }
}

#[cfg(test)]
mod tests {
    use crate::core::contracts::ArrayRegistry;
    use crate::core::errors::SyncWriteError;
    use crate::core::types::{ArrayRegistration, CoordMap};

    use super::InMemoryArrayRegistry;

    #[test]
    fn register_once_and_validate_known_names() {
        let registry = InMemoryArrayRegistry::new();
        registry
            .register(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string(), "pressure".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("registration should succeed");

        registry
            .validate_write_array_names(&["temperature".to_string()])
            .expect("known names should validate");
    }

    #[test]
    fn rejects_second_registration() {
        let registry = InMemoryArrayRegistry::new();
        registry
            .register(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("first registration should succeed");

        let second = registry.register(ArrayRegistration {
            coords: CoordMap::new(),
            array_names: vec!["pressure".to_string()],
            array_dtypes: Vec::new(),
        });
        assert!(matches!(
            second,
            Err(SyncWriteError::ContractViolation { message })
            if message.contains("only be called once")
        ));
    }

    #[test]
    fn rejects_write_validation_before_registration() {
        let registry = InMemoryArrayRegistry::new();
        let result = registry.validate_write_array_names(&["temperature".to_string()]);
        assert!(matches!(
            result,
            Err(SyncWriteError::ContractViolation { message })
            if message.contains("before add_array")
        ));
    }

    #[test]
    fn rejects_unknown_write_array_name() {
        let registry = InMemoryArrayRegistry::new();
        registry
            .register(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("registration should succeed");

        let result = registry.validate_write_array_names(&["humidity".to_string()]);
        assert!(matches!(
            result,
            Err(SyncWriteError::ContractViolation { message })
            if message.contains("unknown array name")
        ));
    }

    #[test]
    fn rejects_empty_coord_axis_before_registration() {
        let mut coords = CoordMap::new();
        let err = coords
            .insert(
                "time".to_string(),
                crate::core::types::CoordValues::I64(vec![]),
            )
            .expect_err("empty coord axis should be rejected when building CoordMap");
        assert!(matches!(
            err,
            SyncWriteError::Validation { message }
            if message.contains("coordinate 'time'") && message.contains("at least one value")
        ));
        coords
            .insert(
                "ensemble".to_string(),
                crate::core::types::CoordValues::U64(vec![0, 1]),
            )
            .expect("non-empty coord axis should be accepted");

        let registry = InMemoryArrayRegistry::new();

        let result = registry.register(ArrayRegistration {
            coords,
            array_names: vec!["temperature".to_string()],
            array_dtypes: Vec::new(),
        });

        assert!(
            result.is_ok(),
            "registration with valid axes should succeed"
        );
    }

    #[test]
    fn rejects_unsafe_array_names_at_registration() {
        for invalid in ["../escape", "nested/name", ".", "..", "name\\part"] {
            let registry = InMemoryArrayRegistry::new();
            let result = registry.register(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec![invalid.to_string()],
                array_dtypes: Vec::new(),
            });
            assert!(
                matches!(
                    result,
                    Err(SyncWriteError::Validation { message })
                    if message.contains("array name") && message.contains("safe path component")
                ),
                "expected unsafe array name '{invalid}' to be rejected"
            );
        }
    }

    #[test]
    fn rejects_unsafe_coord_names_at_registration() {
        for invalid in ["../time", "dim/name", ".", "..", "coord\\part"] {
            let mut coords = CoordMap::new();
            let _ = coords.insert(
                invalid.to_string(),
                crate::core::types::CoordValues::I64(vec![0]),
            );
            let registry = InMemoryArrayRegistry::new();
            let result = registry.register(ArrayRegistration {
                coords,
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            });
            assert!(
                matches!(
                    result,
                    Err(SyncWriteError::Validation { message })
                    if message.contains("coord name") && message.contains("safe path component")
                ),
                "expected unsafe coord name '{invalid}' to be rejected"
            );
        }
    }

    #[test]
    fn accepts_non_empty_coord_axes_at_registration() {
        let mut coords = CoordMap::new();
        let _ = coords.insert(
            "time".to_string(),
            crate::core::types::CoordValues::I64(vec![1, 2, 3]),
        );
        let registry = InMemoryArrayRegistry::new();

        registry
            .register(ArrayRegistration {
                coords,
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("non-empty coordinate axes should be accepted");
    }

    #[test]
    fn resolves_stable_array_ids_from_registration_order() {
        let registry = InMemoryArrayRegistry::new();
        registry
            .register(ArrayRegistration {
                coords: CoordMap::new(),
                array_names: vec![
                    "temperature".to_string(),
                    "pressure".to_string(),
                    "humidity".to_string(),
                ],
                array_dtypes: Vec::new(),
            })
            .expect("registration should succeed");

        let ids = registry
            .resolve_array_ids(&[
                "humidity".to_string(),
                "temperature".to_string(),
                "pressure".to_string(),
            ])
            .expect("array id resolution should succeed");
        assert_eq!(ids, vec![2, 0, 1]);
    }

    #[test]
    fn returns_registered_coords_snapshot() {
        let mut coords = CoordMap::new();
        let _ = coords.insert(
            "time".to_string(),
            crate::core::types::CoordValues::I64(vec![1, 2]),
        );

        let registry = InMemoryArrayRegistry::new();
        registry
            .register(ArrayRegistration {
                coords: coords.clone(),
                array_names: vec!["temperature".to_string()],
                array_dtypes: Vec::new(),
            })
            .expect("registration should succeed");

        let snapshot = registry
            .registered_coords()
            .expect("registered coords should be available");
        assert_eq!(snapshot, coords);
    }
}
