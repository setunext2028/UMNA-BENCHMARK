import math
import unittest

from runtime.abi import (
    Dependency,
    Manifest,
    ResourceCeilings,
    TypeExpr,
    TypeSet,
    WingRef,
)
from runtime.registry import (
    HostMetrics,
    InMemoryRegistry,
    ManifestError,
    VersionError,
    load_manifest,
    parse_manifest,
    version_in_range,
)


def manifest(
    name: str,
    version: str = "1.0.0",
    epoch: str = "1.x",
    inputs=("Text",),
    outputs=("Summary",),
    deps=(),
    max_input_size=100000,
    max_memory_mb=128,
) -> Manifest:
    dep_list = []
    for d in deps:
        if isinstance(d, tuple):
            dep_list.append(Dependency(d[0], d[1]))
        else:
            dep_list.append(d)
    return Manifest(
        name=name,
        version=version,
        core_epoch=epoch,
        author="test",
        input_types=TypeSet(tuple(TypeExpr("Latent", (arg,)) for arg in inputs)),
        output_types=TypeSet(tuple(TypeExpr("Latent", (arg,)) for arg in outputs)),
        dependencies=tuple(dep_list),
        resource_ceilings=ResourceCeilings(max_memory_mb, max_input_size),
    )


BAD_MANIFEST = "not a mapping"
UNKNOWN_KEY = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
unknown_field: "bad"
"""
NESTED_UNKNOWN = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
capabilities:
  - unknown_nested: "bad"
"""
NESTED_TOP_LEVEL = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
unknown_top: "bad"
"""
HOST_OWNED = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
reliability: 0.99
"""
NESTED_HOST_OWNED = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
capabilities:
  - reliability: 0.99
"""
DENY_LIST_BYPASS = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
new_host_field: "bad"
"""
BAD_VERSION = """
wing:
  name: test
  version: "not semver"
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
"""
BAD_RANGE = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
dependencies:
  required:
    - name: alpha
      version_range: "bad range"
"""
BAD_PROVENANCE = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
provenance:
  training_dataset_ids: "not a list"
"""
BAD_CAPABILITY = """
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
capabilities: [123]
"""


class TestManifestParsing(unittest.TestCase):
    def test_valid_manifest_parses(self):
        load_manifest("wings/alpha/manifest.yaml")

    def test_manifest_must_be_mapping(self):
        with self.assertRaises(ManifestError):
            parse_manifest(BAD_MANIFEST)

    def test_unknown_key_is_rejected_by_allow_list(self):
        with self.assertRaises(ManifestError):
            parse_manifest(UNKNOWN_KEY)

    def test_nested_unknown_key_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(NESTED_UNKNOWN)

    def test_nested_key_at_top_level_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(NESTED_TOP_LEVEL)

    def test_anything_host_owned_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(HOST_OWNED)

    def test_nested_host_owned_key_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(NESTED_HOST_OWNED)

    def test_deny_list_cannot_hide_a_new_name(self):
        with self.assertRaises(ManifestError):
            parse_manifest(DENY_LIST_BYPASS)

    def test_non_semver_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(BAD_VERSION)

    def test_malformed_range_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(BAD_RANGE)

    def test_wildcard(self):
        m = load_manifest("wings/alpha/manifest.yaml")
        self.assertEqual(m.dependencies, ())

    def test_exact(self):
        dep = Dependency("foo", "1.2.3")
        self.assertTrue(version_in_range("1.2.3", dep.version_range))
        self.assertFalse(version_in_range("1.2.4", dep.version_range))

    def test_inequalities(self):
        dep = Dependency("foo", ">=1,<3")
        self.assertTrue(version_in_range("1.0.0", dep.version_range))
        self.assertTrue(version_in_range("2.9.9", dep.version_range))
        self.assertFalse(version_in_range("3.0.0", dep.version_range))

    def test_provenance_ids_must_be_a_list(self):
        with self.assertRaises(ManifestError):
            parse_manifest(BAD_PROVENANCE)

    def test_non_string_capability_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(BAD_CAPABILITY)

    def test_reliability_declared_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(HOST_OWNED)

    def test_bad_ceiling_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest("""
wing:
  name: test
  version: 1.0.0
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Summary>"]
resource_ceilings:
  max_memory_mb: "not int"
""")


class TestRegistryLoad(unittest.TestCase):
    def test_load_from_disk_manifest(self):
        registry = InMemoryRegistry(core_epoch="1.x")
        ref = registry.load(load_manifest("wings/alpha/manifest.yaml"))
        self.assertEqual(ref.name, "alpha")
        self.assertEqual(ref.version, "1.0.0")

    def test_load_with_artifact(self):
        registry = InMemoryRegistry(core_epoch="1.x")
        class DummyWing:
            def manifest(self):
                return manifest("dummy")
            def input_types(self):
                return TypeSet((TypeExpr("Latent", ("Text",)),))
            def output_types(self):
                return TypeSet((TypeExpr("Latent", ("Summary",)),))
        wing = DummyWing()
        ref = registry.load(wing.manifest(), wing)
        self.assertIs(registry.artifact(ref), wing)

    def test_incompatible_epoch_rejected_at_load(self):
        registry = InMemoryRegistry(core_epoch="1.x")
        with self.assertRaises(ManifestError):
            registry.load(manifest("w", epoch="9.x"))

    def test_mismatched_epoch_rejected_after_pin(self):
        registry = InMemoryRegistry()
        registry.load(manifest("w"))
        with self.assertRaises(ManifestError):
            registry.load(manifest("w", epoch="9.x"))

    def test_same_ref_with_changed_manifest_is_rejected(self):
        registry = InMemoryRegistry(core_epoch="1.x")
        registry.load(manifest("w"))
        with self.assertRaises(ManifestError):
            registry.load(manifest("w", outputs=("Other",)))

    def test_epoch_cannot_be_repinned(self):
        registry = InMemoryRegistry()
        registry.load(manifest("w"))
        with self.assertRaises(ManifestError):
            registry.pin("9.x")

    def test_pin_rejects_an_incompatible_installed_wing(self):
        registry = InMemoryRegistry()
        registry.load(manifest("w"))
        with self.assertRaises(ManifestError):
            registry.pin("9.x")

    def test_seed_dir_installs_all(self):
        registry = InMemoryRegistry(core_epoch="1.x", seed_dir="wings")
        names = {ref.name for ref in registry.installed()}
        self.assertEqual(len(names), 6)
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertIn("gamma", names)
        self.assertIn("delta", names)
        self.assertIn("epsilon", names)
        self.assertIn("zeta", names)


class TestResolve(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry(core_epoch="1.x")
        for m in (
            manifest("alpha", outputs=("Summary",)),
            manifest("beta", outputs=("Summary",), deps=[("alpha", ">=1,<2")]),
            manifest("gamma", inputs=("Summary",), outputs=("Claim",), deps=[("beta", ">=1,<2")]),
        ):
            self.registry.load(m)

    def test_resolve_brings_dependencies(self):
        resolved = self.registry.resolve([WingRef("beta", "1.0.0")])
        self.assertEqual(resolved, [WingRef("beta", "1.0.0"), WingRef("alpha", "1.0.0")])

    def test_resolve_full_chain(self):
        resolved = self.registry.resolve([WingRef("gamma", "1.0.0")])
        names = [r.name for r in resolved]
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertIn("gamma", names)

    def test_epoch_mismatch_at_resolve(self):
        registry = InMemoryRegistry()
        registry.load(manifest("w", epoch="9.x"))
        with self.assertRaises(VersionError) as ctx:
            registry.resolve([WingRef("w", "1.0.0")], core_epoch="1.x")
        self.assertIn("incompatible Core epoch", str(ctx.exception))

    def test_diamond_refused(self):
        registry = InMemoryRegistry(core_epoch="1.x")
        registry.load(manifest("same", "1.0.0"))
        registry.load(manifest("same", "2.0.0"))
        with self.assertRaises(VersionError) as ctx:
            registry.resolve([WingRef("same", "1.0.0"), WingRef("same", "2.0.0")])
        self.assertIn("diamond", str(ctx.exception))

    def test_unsatisfied_dependency_refused(self):
        self.registry.load(
            manifest("loner", "1.0.0",
                     deps=(Dependency("ghost", ">=1,<2"),))
        )
        with self.assertRaises(VersionError):
            self.registry.resolve([WingRef("loner", "1.0.0")])

    def test_dependency_selection_stays_in_requested_epoch(self):
        registry = InMemoryRegistry(core_epoch="1.x")
        registry.load(manifest("w", epoch="1.x"))
        with self.assertRaises(ManifestError):
            registry.load(manifest("w", epoch="9.x"))


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry(core_epoch="1.x")
        for m in (
            manifest("alpha", outputs=("Summary",)),
            manifest("beta", outputs=("Summary",)),
            manifest("gamma", inputs=("Summary",), outputs=("Claim",)),
        ):
            self.registry.load(m)

    def test_typed_retrieval_honours_types(self):
        hits = self.registry.retrieve(TypeExpr("Latent", ("Text",)), m=10)
        names = {ref.name for ref in hits}
        self.assertIn("alpha", names)
        self.assertIn("beta", names)
        self.assertNotIn("gamma", names)

    def test_retrieval_capped_at_m(self):
        hits = self.registry.retrieve(TypeExpr("Latent", ("Text",)), m=1)
        self.assertEqual(len(hits), 1)

    def test_zero_candidate_limit_returns_nothing(self):
        hits = self.registry.retrieve(TypeExpr("Latent", ("Text",)), m=0)
        self.assertEqual(hits, [])

    def test_mismatched_family_returns_nothing(self):
        hits = self.registry.retrieve(TypeExpr("Image", ("XRay",)), m=10)
        self.assertEqual(hits, [])


class TestCeilingsAndMetrics(unittest.TestCase):
    def setUp(self):
        self.registry = InMemoryRegistry(core_epoch="1.x")
        self.registry.load(manifest("w", inputs=("Text",), max_input_size=100))
        self.ref = WingRef("w", "1.0.0")

    def test_measured_wing_returns_host_metrics(self):
        self.registry.set_measured(
            HostMetrics(self.ref, 5.0, 32.0, 0.9, (0.5,), measured_active_flops=1000.0)
        )
        m = self.registry.measured(self.ref)
        self.assertEqual(m.measured_latency_ms, 5.0)
        self.assertEqual(m.measured_memory_mb, 32.0)

    def test_metrics_cannot_attach_to_uninstalled_ref(self):
        with self.assertRaises(ManifestError):
            self.registry.set_measured(HostMetrics(WingRef("ghost", "1.0.0"), 0.0, 1.0, 1.0, ()))

    def test_non_finite_host_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.set_measured(
                HostMetrics(self.ref, math.nan, 1.0, 1.0, (), measured_active_flops=1.0)
            )

    def test_check_input_allows_within_ceiling(self):
        self.registry.set_measured(HostMetrics(self.ref, 0.0, 10.0, 1.0, (), measured_active_flops=1.0))
        self.assertTrue(self.registry.check_input(self.ref, 50))

    def test_check_input_blocks_above_ceiling(self):
        self.registry.set_measured(HostMetrics(self.ref, 0.0, 1.0, 1.0, (), measured_active_flops=1.0))
        self.assertFalse(self.registry.check_input(self.ref, 10000))

    def test_check_input_negative_size_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.check_input(self.ref, -1)

    def test_check_input_allows_without_metrics(self):
        self.assertTrue(self.registry.check_input(self.ref, 50))

    def test_set_measured_validates_finite(self):
        with self.assertRaises(ValueError):
            self.registry.set_measured(
                HostMetrics(self.ref, math.inf, 1.0, 1.0, (), measured_active_flops=1.0)
            )


if __name__ == "__main__":
    unittest.main()
