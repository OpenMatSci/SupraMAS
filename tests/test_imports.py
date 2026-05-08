def test_lightweight_package_imports():
    import supramas
    import supramas.protocol_optimizer
    import supramas.strategy_tree

    assert supramas.__version__ == "0.1.0"

