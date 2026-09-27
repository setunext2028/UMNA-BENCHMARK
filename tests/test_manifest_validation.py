import unittest

from runtime.abi import TypeExpr
from runtime.registry import (
    ManifestError,
    parse_manifest,
    parse_type_expr,
    version_in_range,
)

GOOD_MANIFEST = """
wing:
  name: test_wing
  version: 1.2.3
  core_epoch: "1.x"
  author: umna
types:
  inputs: ["Latent<Text>"]
  outputs: ["Latent<Claim>"]
capabilities: [test]
dependencies:
  required:
    - name: alpha
      version_range: ">=1.0.0,<2"
resource_ceilings:
  max_memory_mb: 128
  max_input_size: 100000
integrity:
  artifact_hash: "abc123"
provenance:
  training_dataset_ids: ["d1"]
  training_code_commit: "c1"
"""


class TestManifestParsing(unittest.TestCase):
    def test_valid_manifest_parses(self):
        manifest = parse_manifest(GOOD_MANIFEST)
        self.assertEqual(manifest.name, "test_wing")
        self.assertEqual(manifest.version, "1.2.3")
        self.assertEqual(manifest.core_epoch, "1.x")
        self.assertEqual(len(manifest.input_types), 1)
        self.assertEqual(manifest.output_types.entries[0], TypeExpr("Latent", ("Claim",)))
        self.assertEqual(manifest.resource_ceilings.max_memory_mb, 128)
        self.assertEqual(manifest.integrity.artifact_hash, "abc123")

    def test_reliability_declared_is_rejected(self):
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(GOOD_MANIFEST + "reliability: 0.99\n")
        self.assertIn("host-owned", str(ctx.exception))
        self.assertIn("reliability", str(ctx.exception))

    def test_anything_host_owned_is_rejected(self):
        for field in ["accuracy", "expected_latency_ms", "expected_evidence_gain",
                      "routing_priority", "quality", "confidence"]:
            with self.assertRaises(ManifestError, msg=field):
                parse_manifest(GOOD_MANIFEST + f"{field}: 1\n")

    def test_unknown_key_is_rejected_by_allow_list(self):
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(GOOD_MANIFEST + "vendor_score: 1\n")
        self.assertIn("allow-list", str(ctx.exception))

    def test_nested_host_owned_key_is_rejected(self):
        text = GOOD_MANIFEST.replace(
            '      version_range: ">=1.0.0,<2"',
            '      version_range: ">=1.0.0,<2"\n      reliability: 0.99',
        )
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(text)
        self.assertIn("host-owned", str(ctx.exception))

    def test_nested_unknown_key_is_rejected(self):
        text = GOOD_MANIFEST.replace(
            '      version_range: ">=1.0.0,<2"',
            '      version_range: ">=1.0.0,<2"\n      vendor_hint: high',
        )
        with self.assertRaises(ManifestError):
            parse_manifest(text)

    def test_nested_key_at_top_level_is_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST + "name: misplaced\n")

    def test_deny_list_cannot_hide_a_new_name(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST + "expected_information_units: 7\n")

    def test_bad_ceiling_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST.replace("max_memory_mb: 128", "max_memory_mb: -4"))

    def test_non_string_capability_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST.replace("[test]", "[test, 7]"))

    def test_provenance_ids_must_be_a_list(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST.replace('["d1"]', '"d1"'))

    def test_non_semver_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST.replace("version: 1.2.3", "version: latest"))

    def test_malformed_range_rejected(self):
        with self.assertRaises(ManifestError):
            parse_manifest(GOOD_MANIFEST.replace("version_range: \">=1.0.0,<2\"",
                                                 "version_range: banana"))

    def test_manifest_must_be_mapping(self):
        with self.assertRaises(ManifestError):
            parse_manifest("- just\n- a\n- list\n")


class TestTypeExpr(unittest.TestCase):
    def test_parses_family_with_args(self):
        self.assertEqual(parse_type_expr("Latent<Text>"), TypeExpr("Latent", ("Text",)))

    def test_parses_bare_family(self):
        self.assertEqual(parse_type_expr("Latent"), TypeExpr("Latent"))

    def test_rejects_garbage(self):
        with self.assertRaises(ManifestError):
            parse_type_expr("not a type!!")


class TestVersionRanges(unittest.TestCase):
    def test_wildcard(self):
        self.assertTrue(version_in_range("1.0.0", "*"))

    def test_exact(self):
        self.assertTrue(version_in_range("1.0.0", "1.0.0"))
        self.assertFalse(version_in_range("1.0.1", "1.0.0"))

    def test_inequalities(self):
        self.assertTrue(version_in_range("2.5.0", ">=2,<3"))
        self.assertFalse(version_in_range("1.9.9", ">=2,<3"))
        self.assertTrue(version_in_range("2.0.0", ">=2,<3"))
        self.assertTrue(version_in_range("3.0.1", ">3,<4"))

    def test_malformed_raises(self):
        with self.assertRaises(ManifestError):
            version_in_range("1.0.0", "not a range")


if __name__ == "__main__":
    unittest.main()
