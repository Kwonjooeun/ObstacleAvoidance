#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
동적 수중환경 장애물 회피 강화학습 모델 학습 스크립트
모든 장애물 개수(0~16)에서 균형 잡힌 성능을 보장하는 모델 훈련

사용법:
    python train.py --mode [standard|curriculum|balanced] --gpu 1 --gpu_ids 1,2 --workers [num_workers]

옵션:
    --mode: 학습 모드 (standard: 일반 학습, curriculum: 커리큘럼 학습, balanced: 균형 잡힌 학습)
    --gpu: GPU 사용 여부 (0: CPU만 사용, 1: GPU 사용)
    --gpu_ids: 사용할 GPU ID (콤마로 구분, 예: "1,2")
    --workers: 병렬 환경 개수
    --iterations: 총 훈련 반복 횟수
    --checkpoint: 이전 체크포인트에서 계속 학습 (선택 사항)
"""
import inspect
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from abc import ABCMeta

import ray
from ray import tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from ray.rllib.utils import deep_update
from ray.tune.utils import flatten_dict
from obstacle_avoidance import ObstacleAvoidanceEnv
from training_visualizer import TrainingVisualizer

# SAC 알고리즘 구성 (표준 버전)
def sac_config_standard(num_obstacles=1, num_workers=4, use_gpu=True, gpu_ids=None):
    """기본 SAC 구성 생성"""
    config = (
        SACConfig()
        .environment(
            env="obstacle_avoidance",
            env_config={
                "max_obstacles": 16,  # 최대 장애물 수
                "current_max_obstacles": num_obstacles,  # 현재 사용할 최대 장애물 수
                "max_steps": 10,  # 경로점 기반으로 에피소드당 최대 스텝 수 감소
                "step_size": 0.05,  # 충돌 검사용 세분화 스텝 크기
                "collision_threshold": 0.05,  # 충돌 판정 임계값
                "goal_threshold": 0.1,  # 목표 도달 판정 임계값
                "obstacle_speed_range": (0.0, 0.03),  # 장애물 속력 범위
                "max_waypoints": 8  # 최대 경로점 개수
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=1,
            num_envs_per_worker=1,
        )
        .training(
            # 기본 학습 매개변수
            train_batch_size=2048,  # 배치 사이즈
            gamma=0.99,  # 감마 (할인율)
            lr=3e-4,  # 학습률
            
            # SAC 특정 매개변수
            tau=0.005,  # 타겟 네트워크 업데이트 비율
            target_entropy="auto",  # 엔트로피 자동 조정
            initial_alpha=1.0,  # 초기 엔트로피 계수
            twin_q=True,  # 두 개의 Q 함수 사용
            
            # 모델 구성
            model={"fcnet_hiddens": [512, 512, 256]},
            q_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [512, 512]},
            policy_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [512, 512]},
            
            # 리플레이 버퍼 설정
            replay_buffer_config={
                "type": "ReplayBuffer",
                "capacity": 500000,
            },
            
            # 훈련 최적화 설정
            optimization_config={
                "actor_learning_rate": 3e-4,
                "critic_learning_rate": 3e-4,
                "entropy_learning_rate": 3e-4,
            },
            grad_clip=40.0,  # 그래디언트 클리핑
            num_steps_sampled_before_learning_starts=1000,  # 학습 시작 전 샘플링 스텝 수
        )
        .framework(framework="torch")
        .resources(
            num_gpus=(1 if use_gpu else 0),
            num_gpus_per_worker=0
        )
    )   
    return config.to_dict()

# 커리큘럼 학습을 위한 SAC 구성
def sac_config_curriculum(num_obstacles=1, num_workers=4, use_gpu=True, gpu_ids=None):
    """커리큘럼 학습을 위한 SAC 구성 생성"""
    # 기본 구성 생성
    config = (
        SACConfig()
        .environment(
            env="obstacle_avoidance",
            env_config={
                "max_obstacles": 16,
                "current_max_obstacles": num_obstacles,
                "max_steps": 10,
                "step_size": 0.05,
                "collision_threshold": 0.05,
                "goal_threshold": 0.1,
                "obstacle_speed_range": (0.0, 0.03),
                "max_waypoints": 8
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=1,
            num_envs_per_worker=1,
        )
        .training(
            # 작은 배치 사이즈 (초기 단계에서 빠른 학습)
            train_batch_size=1024,
            gamma=0.99,
            
            # SAC 특정 매개변수
            tau=0.005,  # 타겟 네트워크 업데이트 비율
            target_entropy="auto",  # 엔트로피 자동 조정
            initial_alpha=1.0,  # 초기 엔트로피 계수 (높을수록 더 많은 탐색)
            twin_q=True,  # 두 개의 Q 함수 사용
            
            # 모델 구성 (초기 단계에서는 더 작은 네트워크)
            model={"fcnet_hiddens": [256, 256]},
            q_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [256, 256]},
            policy_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [256, 256]},
            
            # 리플레이 버퍼 설정
            replay_buffer_config={
                "type": "ReplayBuffer",
                "capacity": 500000,
            },
            
            # 훈련 최적화 설정 (초기 단계에서는 학습률 약간 높임)
            optimization_config={
                "actor_learning_rate": 5e-4,
                "critic_learning_rate": 5e-4,
                "entropy_learning_rate": 5e-4,
            },
            grad_clip=40.0,  # 그래디언트 클리핑
            num_steps_sampled_before_learning_starts=500,  # 초기 단계에서는 빠르게 학습 시작
        )
        .framework(framework="torch")
        .resources(
            num_gpus=(1 if use_gpu else 0),
            num_gpus_per_worker=0
        )
    )    
    return config.to_dict()

# 균형 잡힌 학습을 위한 분포 설정
def create_balanced_obstacle_distribution():
    """모든 장애물 개수에 대해 균등한 학습 기회를 제공하는 분포 생성"""
    distribution = {}
    for i in range(17):  # 0~16개 장애물
        distribution[i] = 1/17  # 균등 확률
    return distribution

# 균형 잡힌 학습 설정
def sac_config_balanced(num_workers=4, use_gpu=True, gpu_ids=None):
    """모든 장애물 개수에 대해 균형 잡힌 SAC 구성 생성"""
    # 장애물 분포 생성
    obstacle_distribution = create_balanced_obstacle_distribution()
    
    # 기본 구성 생성
    config = (
        SACConfig()
        .environment(
            env="obstacle_avoidance",
            env_config={
                "max_obstacles": 16,
                "current_max_obstacles": 16,  # 최대 장애물 수 사용
                "max_steps": 10,
                "step_size": 0.05,
                "collision_threshold": 0.05,
                "goal_threshold": 0.1,
                "obstacle_speed_range": (0.0, 0.03),
                "max_waypoints": 8,
                "obstacle_distribution": obstacle_distribution  # 균형 잡힌 분포 설정
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=1,
            num_envs_per_worker=1,
        )
        .training(
            # 기본 학습 매개변수
            train_batch_size=2048,
            gamma=0.99,
            
            # SAC 특정 매개변수
            tau=0.005,  # 타겟 네트워크 업데이트 비율
            target_entropy="auto",  # 엔트로피 자동 조정
            initial_alpha=1.0,  # 초기 엔트로피 계수
            twin_q=True,  # 두 개의 Q 함수 사용
            
            # 더 큰 네트워크 모델 (복잡한 문제 해결)
            model={"fcnet_hiddens": [512, 512, 512, 256]},
            q_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [512, 512, 512]},
            policy_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [512, 512, 512]},
            
            # 큰 리플레이 버퍼
            replay_buffer_config={
                "type": "ReplayBuffer",
                "capacity": 1000000,
            },
            
            # 훈련 최적화 설정 (더 낮은 학습률로 안정적인 학습)
            optimization_config={
                "actor_learning_rate": 1e-4,
                "critic_learning_rate": 1e-4,
                "entropy_learning_rate": 1e-4,
            },
            grad_clip=40.0,  # 그래디언트 클리핑
            num_steps_sampled_before_learning_starts=2000,  # 더 많은 경험을 쌓은 후 학습 시작
            
            # 추가 훈련 강도 설정
            training_intensity=1.0,
            n_step=1,
        )
        .framework(framework="torch")
        .resources(
            num_gpus=(1 if use_gpu else 0),
            num_gpus_per_worker=0
        )
    )    
    return config.to_dict()

# 주어진 GPU ID를 설정하고 Ray 초기화
def ray_init_with_gpu_filter(gpu_ids=None):
    """GPU ID를 제한하여 Ray 초기화"""
    if not ray.is_initialized():
        # GPU ID가 제공된 경우 환경 변수 설정
        if gpu_ids and len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            print(f"CUDA_VISIBLE_DEVICES 설정: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        ray_params = {
            "include_dashboard": False,
            "ignore_reinit_error": True,
        }
        
        ray.init(**ray_params)

# 장애물 환경 테스트 함수
def test_obstacle_env(env_config):
    """장애물 환경 생성 및 테스트"""
    print("\n=== 장애물 환경 테스트 ===")
    
    # 환경 생성
    env = ObstacleAvoidanceEnv(env_config)
    
    # 장애물 생성 로직 테스트
    env.reset()
    
    # 장애물 정보 출력
    valid_obstacles = [obs for obs in env.obstacles if obs["radius"] > 0]
    print(f"총 장애물 슬롯: {len(env.obstacles)}")
    print(f"유효한 장애물 수: {len(valid_obstacles)}")
    print(f"current_max_obstacles: {env.current_max_obstacles}")
    
    if len(valid_obstacles) > 0:
        print(f"첫 번째 장애물 정보:")
        print(f"  위치: {valid_obstacles[0]['position']}")
        print(f"  반지름: {valid_obstacles[0]['radius']}")
    else:
        print(f"유효한 장애물이 없습니다!")
    
    # 환경 테스트
    for i in range(5):  # 5회 테스트
        # 환경 리셋
        obs, info = env.reset()
        
        # 무작위 액션
        action = env.action_space.sample()
        
        # 액션 실행
        next_obs, reward, done, truncated, info = env.step(action)
        
        # 결과 출력
        print(f"테스트 {i+1}:")
        print(f"  액션: {action}")
        print(f"  보상: {reward}")
        print(f"  종료: {done}")
        print(f"  정보: {info}")
        
    return env

# 표준 학습 실행
def standard_training(args):
    """일반적인 학습 방식 (최대 난이도에서 바로 시작)"""
    # GPU ID 파싱
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')] if args.gpu_ids else []
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정 및 Ray 초기화
    ray_init_with_gpu_filter(gpu_ids if args.gpu else None)
    
    # 결과 디렉토리 설정
    result_dir = f"results/standard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    absolute_path = os.path.abspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    # 설정 저장
    config = sac_config_standard(
        num_obstacles=16, 
        num_workers=args.workers, 
        use_gpu=args.gpu == 1, 
        gpu_ids=gpu_ids if args.gpu else None
    )

    # 설정을 JSON 직렬화 가능한 형태로 변환
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items() 
                    if not (inspect.isfunction(v) or inspect.ismethod(v) or 
                            isinstance(v, ABCMeta) or inspect.isclass(v))}
        elif isinstance(obj, list):
            return [make_json_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            try:
                return str(obj)
            except:
                return None

    # 설정 저장
    serializable_config = make_json_serializable(config)
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    # 학습 시각화 설정
    # 500 iterations에서 약 50번 시각화 = 10번에 한 번
    viz_interval = max(1, args.iterations // 50)
    
    # 시각화 콜백 생성
    def env_creator(env_config):
        return ObstacleAvoidanceEnv(env_config)
    
    # 시각화 객체 생성
    visualizer = TrainingVisualizer(
        env_creator=env_creator,
        config=config,
        result_dir=absolute_path,
        visualization_interval=viz_interval,
        max_visualizations=50,
        save_gif=False
    )
    
    # 콜백 함수 설정
    callbacks = {
        "on_train_result": visualizer.on_train_result
    }
    
    # 학습 실행
    print("\n=== 표준 학습 시작 (장애물 16개로 직접 훈련) ===")
    restore_path = args.checkpoint if args.checkpoint else None
    result = tune.run(
        "SAC",
        config=config,
        stop={"training_iteration": args.iterations},
        checkpoint_freq=args.iterations // 10,
        checkpoint_at_end=True,
        name=f"standard_training",
        local_dir=absolute_path,
        restore=restore_path,
        verbose=1,
        callbacks=callbacks
    )
    
    # 최종 체크포인트 저장
    best_checkpoint = result.get_best_checkpoint(
        metric="episode_reward_mean", 
        mode="max"
    )
    
    # 체크포인트 정보 저장
    with open(os.path.join(result_dir, "checkpoint_info.txt"), 'w') as f:
        f.write(f"Best checkpoint: {best_checkpoint}\n")
    
    # 학습 요약 생성
    final_result = result.trials[0].last_result
    visualizer.create_training_summary(final_result)
    
    ray.shutdown()
    return best_checkpoint

# 커리큘럼 학습 실행
def curriculum_training(args):
    """커리큘럼 학습 진행 (점진적으로 장애물 수 증가)"""
    # GPU ID 파싱 및 초기화
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')] if args.gpu_ids else []
    ray_init_with_gpu_filter(gpu_ids if args.gpu else None)
    
    # 결과 디렉토리 설정
    result_dir = f"results/curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    absolute_path = os.path.abspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    # 초기 체크포인트 경로
    checkpoint_path = args.checkpoint
    
    # 학습 시각화 설정
    # 레벨당 시각화 횟수 계산 (전체적으로 50회가 되도록)
    total_levels = 16  # 1~16개 장애물
    viz_per_level = max(1, 50 // total_levels)
    
    # 환경 생성 함수
    def env_creator(env_config):
        return ObstacleAvoidanceEnv(env_config)
    
    # 최종 결과 저장
    all_results = []
    
    # 장애물 단계별 훈련
    for obstacle_level in range(1, 17):  # 1개부터 16개까지
        print(f"\n=== 커리큘럼 레벨 {obstacle_level}: 장애물 {obstacle_level}개 학습 ===")
        
        # 환경 구성
        config = sac_config_curriculum(
            num_obstacles=obstacle_level,
            num_workers=args.workers, 
            use_gpu=args.gpu == 1, 
            gpu_ids=gpu_ids if args.gpu else None
        )
        
        # 이전 체크포인트에서 훈련 재개
        if checkpoint_path:
            config["restore"] = checkpoint_path
        
        # 레벨별 디렉토리
        level_dir = os.path.join(result_dir, f"level_{obstacle_level}")
        os.makedirs(level_dir, exist_ok=True)
        
        # 시각화 콜백 생성
        visualizer_callback = create_visualizer_callback(
            env_creator=env_creator,
            env_config=config["env_config"],
            result_dir=level_dir,
            viz_interval=max(1, args.iterations // 16 // viz_per_level),
            max_viz=viz_per_level,
            save_gif=False
        )
        
        # 훈련 실행
        iterations_per_level = max(args.iterations // 16, 20)  # 레벨당 최소 20회 반복
        result = tune.run(
            "SAC",
            config=config,
            stop={"training_iteration": iterations_per_level},
            checkpoint_freq=iterations_per_level // 2,
            checkpoint_at_end=True,
            name=f"curriculum_level_{obstacle_level}",
            local_dir=absolute_path,
            restore=checkpoint_path,
            verbose=1,
            callbacks=[visualizer_callback]  # 수정: 리스트로 전달
        )
        
        # 최고 체크포인트 갱신
        checkpoint_path = result.get_best_checkpoint(
            metric="episode_reward_mean", 
            mode="max"
        )
        
        # 중간 결과 저장
        with open(os.path.join(level_dir, f"level_{obstacle_level}_checkpoint.txt"), 'w') as f:
            f.write(f"Level {obstacle_level} best checkpoint: {checkpoint_path}\n")
        
        # 결과 저장
        final_result = result.trials[0].last_result
        all_results.append({
            "level": obstacle_level,
            "reward_mean": final_result.get("episode_reward_mean", 0),
            "success_rate": final_result.get("custom_metrics", {}).get("success_rate", 0)
        })
    
    # 모든 레벨 결과 시각화
    plt.figure(figsize=(12, 8))
    
    # 보상 그래프
    plt.subplot(2, 1, 1)
    plt.plot([r["level"] for r in all_results], [r["reward_mean"] for r in all_results], 'b-o', linewidth=2)
    plt.xlabel('장애물 개수')
    plt.ylabel('평균 보상')
    plt.title('커리큘럼 학습: 레벨별 평균 보상')
    plt.grid(True)
    
    # 성공률 그래프
    plt.subplot(2, 1, 2)
    plt.plot([r["level"] for r in all_results], [r.get("success_rate", 0) for r in all_results], 'g-o', linewidth=2)
    plt.xlabel('장애물 개수')
    plt.ylabel('성공률 (%)')
    plt.title('커리큘럼 학습: 레벨별 성공률')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "curriculum_learning_progress.png"), dpi=150)
    plt.close()
    
    # 최종 체크포인트 저장
    with open(os.path.join(result_dir, "final_checkpoint.txt"), 'w') as f:
        f.write(f"Final best checkpoint: {checkpoint_path}\n")
    
    ray.shutdown()
    return checkpoint_path

# 균형 잡힌 학습 실행
def balanced_training(args):
    """균형 잡힌 학습 방식 (모든 장애물 개수에 대해 균등한 학습 기회)"""
    # GPU ID 파싱
    gpu_ids = [int(id.strip()) for id in args.gpu_ids.split(',')] if args.gpu_ids else []
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정 및 Ray 초기화
    ray_init_with_gpu_filter(gpu_ids if args.gpu else None)
    
    # 결과 디렉토리 설정
    result_dir = f"results/balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    absolute_path = os.path.abspath(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    # 설정 저장
    config = sac_config_balanced(
        num_workers=args.workers, 
        use_gpu=args.gpu == 1, 
        gpu_ids=gpu_ids if args.gpu else None
    )

    # 설정을 JSON 직렬화 가능한 형태로 변환
    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items() 
                    if not (inspect.isfunction(v) or inspect.ismethod(v) or 
                            isinstance(v, ABCMeta) or inspect.isclass(v))}
        elif isinstance(obj, list):
            return [make_json_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            try:
                return str(obj)
            except:
                return None

    # 설정 저장
    serializable_config = make_json_serializable(config)
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    # 학습 시각화 설정
    # 500 iterations에서 약 50번 시각화 = 10번에 한 번
    viz_interval = max(1, args.iterations // 50)
    
    # 환경 생성 함수
    def env_creator(env_config):
        return ObstacleAvoidanceEnv(env_config)
    
    # 시각화 콜백 생성
    visualizer_callback = create_visualizer_callback(
        env_creator=env_creator,
        env_config=config["env_config"],
        result_dir=result_dir,
        viz_interval=viz_interval,
        max_viz=50,
        save_gif=True  # 균형 잡힌 학습은 최종 모델이므로 GIF 생성
    )
    
    # 학습 실행
    print("\n=== 균형 잡힌 학습 시작 (모든 장애물 개수에 대해 균등 학습) ===")
    restore_path = args.checkpoint if args.checkpoint else None
    result = tune.run(
        "SAC",
        config=config,
        stop={"training_iteration": args.iterations},
        checkpoint_freq=args.iterations // 10,
        checkpoint_at_end=True,
        name=f"balanced_training",
        local_dir=absolute_path,
        restore=restore_path,
        verbose=1,
        callbacks=[visualizer_callback]  # 수정: 리스트로 전달
    )
    
    # 최종 체크포인트 저장
    best_checkpoint = result.get_best_checkpoint(
        metric="episode_reward_mean", 
        mode="max"
    )
    
    # 체크포인트 정보 저장
    with open(os.path.join(result_dir, "checkpoint_info.txt"), 'w') as f:
        f.write(f"Best checkpoint: {best_checkpoint}\n")
    
    ray.shutdown()
    return best_checkpoint

# 메인 함수
def main():
    # 인자 파서 설정
    parser = argparse.ArgumentParser(description='동적 수중환경 장애물 회피 강화학습 모델 학습')
    parser.add_argument('--mode', type=str, default='balanced', choices=['standard', 'curriculum', 'balanced', 'test'],
                        help='학습 모드 (standard/curriculum/balanced/test)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 사용 여부 (0: CPU만, 1: GPU 사용)')
    parser.add_argument('--gpu_ids', type=str, default="1,2", help='사용할 GPU ID (콤마로 구분, 예: "1,2")')
    parser.add_argument('--workers', type=int, default=4, help='병렬 환경 개수')
    parser.add_argument('--iterations', type=int, default=500, help='총 훈련 반복 횟수')
    parser.add_argument('--checkpoint', type=str, default=None, help='이전 체크포인트에서 계속 학습 (선택 사항)')
    parser.add_argument('--num_obstacles', type=int, default=16, help='장애물 개수 (테스트 모드 시 사용)')
    
    args = parser.parse_args()
    
    # 시작 메시지
    print(f"\n=== 동적 수중환경 장애물 회피 강화학습 모델 학습 시작 ===")
    print(f"모드: {args.mode}")
    print(f"GPU 사용: {'예' if args.gpu else '아니오'}")
    if args.gpu:
        print(f"사용 GPU ID: {args.gpu_ids}")
    print(f"병렬 환경 개수: {args.workers}")
    print(f"총 반복 횟수: {args.iterations}")
    if args.checkpoint:
        print(f"이전 체크포인트: {args.checkpoint}")
    
    # 환경을 ray에 등록
    def env_creator(env_config):
        return ObstacleAvoidanceEnv(env_config)

    register_env("obstacle_avoidance", env_creator)
    
    # 테스트 모드
    if args.mode == 'test':
        # 간단한 환경 테스트
        env_config = {
            "max_obstacles": 16,
            "current_max_obstacles": args.num_obstacles,
            "max_steps": 10,
            "step_size": 0.05,
            "collision_threshold": 0.05,
            "goal_threshold": 0.1,
            "obstacle_speed_range": (0.0, 0.03),
            "max_waypoints": 8
        }
        test_obstacle_env(env_config)
        return
    
    # 선택한 모드에 따라 학습 진행
    if args.mode == 'standard':
        checkpoint = standard_training(args)
    elif args.mode == 'curriculum':
        checkpoint = curriculum_training(args)
    else:  # balanced
        checkpoint = balanced_training(args)
    
    print(f"\n훈련 완료! 최종 체크포인트: {checkpoint}")
    print("모델 평가를 위해 다음 명령을 실행하세요:")
    print(f"python evaluation.py --checkpoint {checkpoint} --gpu {args.gpu} --gpu_ids {args.gpu_ids}")


if __name__ == "__main__":
    main()
