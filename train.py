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
                "max_steps": 100,  # 에피소드 최대 스텝 수
                "step_size": 0.05,  # 한 스텝당 이동 거리
                "collision_threshold": 0.05,  # 충돌 판정 임계값
                "goal_threshold": 0.1,  # 목표 도달 판정 임계값
                "obstacle_speed_range": (0.0, 0.03)  # 장애물 속력 범위
            },
        )
        .rollouts(
            num_rollout_workers=num_workers,
            rollout_fragment_length=1,
            num_envs_per_worker=1,
        )
        .training(
            train_batch_size=2048,  # 배치 사이즈
            gamma=0.99,  # 감마 (할인율)
            lr=3e-4,  # 학습률
            tau=0.005,  # 타겟 네트워크 업데이트 비율
            # SAC 특정 파라미터 - 직접 설정
            #optimization={"actor_learning_rate": 1e-3, "critic_learning_rate": 1e-3, "entropy_learning_rate": 1e-3},
            # 신경망 크기 설정
            model={"fcnet_hiddens": [256, 256]},
            # 리플레이 버퍼 설정
            # replay_buffer_config={
            #     "type": "MultiAgentReplayBuffer",
            #     "capacity": 100000,
            # },
            # # 우선순위 리플레이 설정 - 필요한 모든 매개변수 포함
            # prioritized_replay=True,
            # prioritized_replay_alpha=0.6,
            # prioritized_replay_beta=0.4,  # 중요: beta 값 명시적 설정
            # prioritized_replay_eps=1e-6
        )
        .framework(framework="torch")
        .resources(
            num_gpus=(1 if use_gpu else 0),
            num_gpus_per_worker=0
        )
    )
    # print(inspect.signature(config.experimental))
    # 리플레이 버퍼 설정 - 직접 메서드로 설정
    # 방법 1: experimental() 메서드를 사용하되 인자를 키워드로 전달
    # 경고도 표시되었으니 lz4 설치 권장
    # config = config.experimental(
    #     replay_buffer_config={
    #         "type": "ReplayBuffer",  # 일반 리플레이 버퍼 사용
    #         "capacity": 100000
    #     }
    # )


    # # 사용 가능한 GPU 제한 설정
    # if use_gpu and gpu_ids and len(gpu_ids) > 0:
    #     config.update({
    #         "_fake_gpus": True, # 특정 GPU 강제 지정을 위해 필요
    #         # 실제 구현에서는 보통 이렇게 설정함
    #         "_available_gpus": [0],  # 0으로 설정해도 위에서 CUDA_VISIBLE_DEVICES로 제한된 GPU를 사용하게 됨
    #     })
    
    return config.to_dict()# 주어진 GPU ID를 설정하고 Ray 초기화
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
    file_uri = f"file://{absolute_path}"
    os.makedirs(result_dir, exist_ok=True)
    # 설정 저장
    config = sac_config_standard(
        num_obstacles=16, 
        num_workers=args.workers, 
        use_gpu=args.gpu == 1, 
        gpu_ids=gpu_ids if args.gpu else None
    )

    # with open(os.path.join(result_dir, "config.json"), 'w') as f:
    #     json.dump(config, f, indent=4)

    # 설정을 평면화된 딕셔너리로 변환
    # flat_config = {}
    # if hasattr(config, "to_dict"):
    #   flat_config = flatten_dict(config.to_dict())
    print(config)
    # flat_config = flatten_dict(config.to_dict())
    # else:
    #     # 수동으로 설정 값 추출
    #     flat_config = {
    #         "algorithm": "SAC",
    #         # 다른 기본 정보 추가
    #     }

    # JSON으로 저장
    # with open(os.path.join(result_dir, "config.json"), 'w') as f:
    #     json.dump(flat_config, f, indent=4)

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

    # 사용 예시
    serializable_config = make_json_serializable(config)
    with open(os.path.join(result_dir, "config.json"), 'w') as f:
        json.dump(serializable_config, f, indent=4)
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
        local_dir=file_uri,
        restore=restore_path,
        verbose=1
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
    parser.add_argument('--mode', type=str, default='balanced', choices=['standard', 'curriculum', 'balanced'],
                        help='학습 모드 (standard/curriculum/balanced)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU 사용 여부 (0: CPU만, 1: GPU 사용)')
    parser.add_argument('--gpu_ids', type=str, default="1,2", help='사용할 GPU ID (콤마로 구분, 예: "1,2")')
    parser.add_argument('--workers', type=int, default=4, help='병렬 환경 개수')
    parser.add_argument('--iterations', type=int, default=500, help='총 훈련 반복 횟수')
    parser.add_argument('--checkpoint', type=str, default=None, help='이전 체크포인트에서 계속 학습 (선택 사항)')
    
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
    
    # 선택한 모드에 따라 학습 진행
    if args.mode == 'standard':
        # 환경을 ray에 등록
        def env_creator(env_config):
            return ObstacleAvoidanceEnv(env_config)

        register_env("obstacle_avoidance", env_creator)
        checkpoint = standard_training(args)
    # elif args.mode == 'curriculum':
    #     checkpoint = curriculum_training(args)
    # else:  # balanced
    #     checkpoint = balanced_training(args)
    
    print(f"\n훈련 완료! 최종 체크포인트: {checkpoint}")
    print("모델 평가를 위해 다음 명령을 실행하세요:")
    print(f"python evaluation.py --checkpoint {checkpoint} --gpu 1 --gpu_ids {args.gpu_ids}")


if __name__ == "__main__":
    main()
