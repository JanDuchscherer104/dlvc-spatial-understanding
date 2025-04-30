import unittest

from ..utils.utils import BaseConfig


class ChildConfig(BaseConfig[None]):
    foo: int = 0


class ParentConfig(BaseConfig[None]):
    foo: int = 1
    child: ChildConfig = ChildConfig()
    protected_fields: list = ["foo"]


class TestBaseConfig(unittest.TestCase):
    def test_protected_fields_not_propagated(self):
        parent = ParentConfig()
        child = parent.child
        # protected_fields includes 'foo', so foo should not be propagated
        self.assertEqual(child.foo, 0)
        self.assertNotIn("foo", child.propagated_fields)

    def test_non_protected_fields_propagated(self):
        for value in ["x", "y"]:
            # Dynamically set bar on parent and expect propagation
            pc = ParentConfig2(bar=value)
            child = pc.child
            self.assertEqual(child.bar, value)
            self.assertIn("bar", child.propagated_fields)


class ParentConfig2(BaseConfig[None]):
    bar: str = "parent"
    child: ChildConfig = ChildConfig()
    protected_fields: list = []


if __name__ == "__main__":
    unittest.main()
