def pytest_collection_modifyitems(config, items):
    items[:] = [
        cur_item for cur_item in items if cur_item.name != 'test_session']
