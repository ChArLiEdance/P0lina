import imp


def make_network(cfg):
    """
    根据配置创建网络
    支持动态选择网络类型
    """
    module = cfg.network_module
    path = cfg.network_path
    
    # 检查是否有网络类型参数
    network_type = getattr(cfg.task_arg, 'network_type', 'nerf')
    
    print(f"Creating network with type: {network_type}")
    print(f"Network module: {module}")
    print(f"Network path: {path}")
    
    # 动态加载网络模块
    network = imp.load_source(module, path).Network()
    
    # 打印网络信息
    if hasattr(network, 'network_type'):
        print(f"Network type: {network.network_type}")
    
    return network
