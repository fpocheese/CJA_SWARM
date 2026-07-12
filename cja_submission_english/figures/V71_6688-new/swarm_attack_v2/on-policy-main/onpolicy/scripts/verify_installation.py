#!/usr/bin/env python
"""
代码验证脚本 - 检查所有改进模块是否可以正常导入
"""
import sys
import traceback

def test_imports():
    """测试所有模块导入"""
    print("="*80)
    print("开始验证改进版MAPPO代码...")
    print("="*80 + "\n")
    
    tests = []
    
    # 测试1: 导入配置
    print("[1/6] 测试配置模块...")
    try:
        from onpolicy.config_advanced import get_advanced_config
        config = get_advanced_config()
        print("✓ 配置模块导入成功")
        tests.append(True)
    except Exception as e:
        print(f"✗ 配置模块导入失败: {e}")
        traceback.print_exc()
        tests.append(False)
    
    # 测试2: 导入改进的Actor-Critic
    print("\n[2/6] 测试改进的Actor-Critic网络...")
    try:
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_advanced import (
            MultiHeadAttention, ResidualBlock, R_Actor_Advanced, R_Critic_Advanced
        )
        print("✓ Actor-Critic模块导入成功")
        print("  - MultiHeadAttention: ✓")
        print("  - ResidualBlock: ✓")
        print("  - R_Actor_Advanced: ✓")
        print("  - R_Critic_Advanced: ✓")
        tests.append(True)
    except Exception as e:
        print(f"✗ Actor-Critic模块导入失败: {e}")
        traceback.print_exc()
        tests.append(False)
    
    # 测试3: 导入改进的Policy
    print("\n[3/6] 测试改进的Policy...")
    try:
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_advanced import R_MAPPOPolicy_Advanced
        print("✓ Policy模块导入成功")
        tests.append(True)
    except Exception as e:
        print(f"✗ Policy模块导入失败: {e}")
        traceback.print_exc()
        tests.append(False)
    
    # 测试4: 导入改进的训练算法
    print("\n[4/6] 测试改进的训练算法...")
    try:
        from onpolicy.algorithms.r_mappo.r_mappo_advanced import R_MAPPO_Advanced
        print("✓ 训练算法模块导入成功")
        tests.append(True)
    except Exception as e:
        print(f"✗ 训练算法模块导入失败: {e}")
        traceback.print_exc()
        tests.append(False)
    
    # 测试5: 导入改进的Runner
    print("\n[5/6] 测试改进的Runner...")
    try:
        from onpolicy.runner.shared.mpe_runner_advanced import MPERunner
        print("✓ Runner模块导入成功")
        tests.append(True)
    except Exception as e:
        print(f"✗ Runner模块导入失败: {e}")
        traceback.print_exc()
        tests.append(False)
    
    # 测试6: 测试网络实例化
    print("\n[6/6] 测试网络实例化...")
    try:
        import torch
        import argparse
        from gym import spaces
        
        # 创建简单的参数
        args = argparse.Namespace(
            hidden_size=64,
            gain=0.01,
            use_orthogonal=True,
            use_policy_active_masks=True,
            use_naive_recurrent_policy=False,
            use_recurrent_policy=True,
            recurrent_N=1,
            use_attention=True,
            use_residual=True,
            use_popart=False
        )
        
        # 创建简单的空间
        obs_space = spaces.Box(low=-1, high=1, shape=(10,))
        action_space = spaces.Discrete(5)
        
        # 实例化Actor
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_advanced import R_Actor_Advanced
        actor = R_Actor_Advanced(args, obs_space, action_space)
        print("✓ Actor实例化成功")
        
        # 实例化Critic
        from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_advanced import R_Critic_Advanced
        critic = R_Critic_Advanced(args, obs_space)
        print("✓ Critic实例化成功")
        
        # 测试前向传播
        obs = torch.randn(1, 10)
        rnn_states = torch.zeros(1, 1, 64)
        masks = torch.ones(1, 1)
        
        actions, log_probs, new_rnn_states = actor(obs, rnn_states, masks)
        print("✓ Actor前向传播成功")
        
        values, new_rnn_states = critic(obs, rnn_states, masks)
        print("✓ Critic前向传播成功")
        
        tests.append(True)
    except Exception as e:
        print(f"✗ 网络实例化失败: {e}")
        traceback.print_exc()
        tests.append(False)
    
    # 总结
    print("\n" + "="*80)
    print("验证结果:")
    print("="*80)
    passed = sum(tests)
    total = len(tests)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n✓ 所有测试通过！代码可以正常使用。")
        print("\n下一步:")
        print("1. 修改 scripts/train_mpe_advanced.sh 中的参数")
        print("2. 运行训练: cd scripts && ./train_mpe_advanced.sh")
        return True
    else:
        print(f"\n✗ {total - passed} 个测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
