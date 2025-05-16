#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
학습 과정 시각화 및 모니터링 기능

이 모듈은 강화학습 훈련 과정에서 환경과 에이전트의 행동을 주기적으로 시각화하는
기능을 제공합니다. 학습 상태를 3D 플롯으로 저장하여 성능을 모니터링할 수 있습니다.

사용법:
    train.py에서 import하여 사용합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import datetime
import json
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.logger import pretty_print
from ray.tune.callback import Callback

class TrainingVisualizerCallback(Callback):
    """Ray Tune 호환 시각화 콜백 클래스"""
    
    def __init__(self, env_creator, env_config, result_dir, visualization_interval=10, 
                 max_visualizations=50, save_gif=False):
        """
        Args:
            env_creator: 환경 생성 함수
            env_config: 환경 구성
            result_dir: 결과 저장 디렉토리
            visualization_interval: 시각화 간격 (iterations)
            max_visualizations: 최대 시각화 횟수
            save_gif: GIF 저장 여부
        """
        self.env_creator = env_creator
        self.env_config = env_config
        self.result_dir = result_dir
        self.visualization_interval = visualization_interval
        self.max_visualizations = max_visualizations
        self.save_gif = save_gif
        
        # 시각화 저장 디렉토리
        self.viz_dir = os.path.join(result_dir, "visualizations")
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 평가 환경 생성
        self.env = env_creator(env_config)
        
        # 시각화 카운터
        self.viz_count = 0
        
        # 학습 진행 정보 저장 (보상 기록)
        self.rewards_history = []
        
        print(f"학습 시각화 설정 완료: 간격={visualization_interval}, 최대={max_visualizations}")

    def on_trial_result(self, iteration, trials, trial, result, **info):
        """학습 결과를 받을 때마다 호출되는 콜백"""
        iteration = result["training_iteration"]
        
        # 보상 이력 저장
        reward_mean = result.get("episode_reward_mean", 0)
        self.rewards_history.append((iteration, reward_mean))
        
        # 시각화 간격에 따라 시각화 수행
        if iteration % self.visualization_interval == 0 and self.viz_count < self.max_visualizations:
            print(f"\n=== {iteration}번째 iteration 시각화 중... ===")
            
            # 환경 리셋 및 장애물 디버깅
            obs, env_info = self.env.reset()
            
            # 장애물 정보 출력 (디버깅)
            valid_obstacles = [obs for obs in self.env.obstacles if obs["radius"] > 0]
            print(f"장애물 정보 디버깅:")
            print(f"  총 장애물 슬롯: {len(self.env.obstacles)}")
            print(f"  유효한 장애물 수: {len(valid_obstacles)}")
            print(f"  current_max_obstacles: {self.env.current_max_obstacles}")
            
            if len(valid_obstacles) > 0:
                print(f"  첫 번째 장애물 정보:")
                print(f"    위치: {valid_obstacles[0]['position']}")
                print(f"    반지름: {valid_obstacles[0]['radius']}")
            else:
                print(f"  유효한 장애물이 없습니다!")
            
            # 현재 체크포인트 및 배우기 경로 얻기
            checkpoint_dir = os.path.join(
                os.path.dirname(trial.logdir), 
                os.path.basename(trial.logdir), 
                "checkpoint_000000"
            )
            
            # 현재 정책 상태 시각화 (체크포인트가 없으면 환경만 시각화)
            sample_policy = None
            try:
                if os.path.exists(checkpoint_dir):
                    # 체크포인트에서 정책 로드 시도
                    sample_policy = self.visualize_checkpoint(checkpoint_dir, iteration)
                else:
                    # 체크포인트가 없으면 환경 정보만 시각화
                    self.visualize_env_only(result, iteration)
            except Exception as e:
                print(f"시각화 오류: {e}")
                # 오류 발생 시 환경만 시각화
                self.visualize_env_only(result, iteration)
            
            # 학습 진행 그래프 생성
            self.plot_training_progress(result, iteration)
            
            self.viz_count += 1
            
        # 주요 지표 출력
        print("\n학습 진행 상태:")
        metrics = {
            "iteration": iteration,
            "episode_reward_mean": result.get("episode_reward_mean", 0),
            "episode_reward_min": result.get("episode_reward_min", 0),
            "episode_reward_max": result.get("episode_reward_max", 0),
            "episode_len_mean": result.get("episode_len_mean", 0),
        }
        print(pretty_print(metrics))

    def visualize_checkpoint(self, checkpoint_dir, iteration):
        """체크포인트에서 정책 로드하여 시각화"""
        # 체크포인트 메타데이터 확인
        metadata_path = os.path.join(checkpoint_dir, "checkpoint.json")
        if not os.path.exists(metadata_path):
            print(f"체크포인트 메타데이터를 찾을 수 없음: {metadata_path}")
            return None

        try:
            # 메타데이터를 통해 정확한 체크포인트 경로 확인
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            checkpoint_path = os.path.join(checkpoint_dir, metadata["checkpoint_name"])
            if not os.path.exists(checkpoint_path):
                print(f"체크포인트 파일을 찾을 수 없음: {checkpoint_path}")
                return None
            
            # 모델 로드 시도
            loaded_algorithm = Algorithm.from_checkpoint(checkpoint_path)
            
            # 환경 리셋
            obs, info = self.env.reset()
            
            # 액션 계산
            action = loaded_algorithm.compute_single_action(obs)
            
            # 경로점 추출
            waypoints = self.env._get_waypoints_from_action(action)
            
            # 경로점 시각화
            self.visualize_path(waypoints, iteration, loaded_algorithm)
            
            return loaded_algorithm

        except Exception as e:
            print(f"체크포인트 로드 오류: {e}")
            return None

    def visualize_path(self, waypoints, iteration, algorithm=None):
        """경로점과 환경 시각화"""
        # 결과 저장 경로
        save_path = os.path.join(self.viz_dir, f"iteration_{iteration:04d}.png")
        
        # 3D 시각화
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 시작점 및 목표점 표시
        ax.scatter(self.env.start_position[0], self.env.start_position[1], self.env.start_position[2], 
                color='green', s=200, marker='o', label='Start')
        ax.scatter(self.env.goal_position[0], self.env.goal_position[1], self.env.goal_position[2], 
                color='red', s=200, marker='*', label='Goal')
        
        # 환경에 있는 장애물 시각화
        for i, obstacle in enumerate(self.env.obstacles):
            if obstacle["radius"] > 0:  # 유효한 장애물만
                # 장애물 위치 및 반지름
                pos = obstacle["position"]
                r = obstacle["radius"]
                
                # 실린더 바닥면 그리기 (원)
                theta = np.linspace(0, 2*np.pi, 50)
                x = pos[0] + r * np.cos(theta)
                y = pos[1] + r * np.sin(theta)
                z = np.zeros_like(theta)  # z=0 평면에 그리기
                
                # 실린더 윗면 그리기
                x2 = pos[0] + r * np.cos(theta)
                y2 = pos[1] + r * np.sin(theta)
                z2 = np.ones_like(theta)  # z=1 평면에 그리기
                
                # 실린더 표면 그리기
                ax.plot(x, y, z, 'gray', alpha=0.5)
                ax.plot(x2, y2, z2, 'gray', alpha=0.5)
                
                # 기둥선 그리기 (4개 모서리)
                for j in range(0, 50, 12):
                    ax.plot([x[j], x2[j]], [y[j], y2[j]], [z[j], z2[j]], 'gray', alpha=0.5, linewidth=0.7)
                
                # 장애물 ID 표시
                ax.text(pos[0], pos[1], pos[2] + 0.1, f"Obs{i+1}", color='black', fontsize=8)
                
                # 움직이는 장애물인 경우 방향 표시
                if obstacle["speed"] > 0:
                    dx = 0.2 * np.cos(obstacle["course"])
                    dy = 0.2 * np.sin(obstacle["course"])
                    ax.quiver(pos[0], pos[1], pos[2], dx, dy, 0, color='red', arrow_length_ratio=0.3)
        
        # 경로점 시각화
        status_text = []
        if waypoints:
            # 경로점 배열 변환
            waypoints_array = np.array(waypoints)
            
            # 경로점 표시
            ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], waypoints_array[:, 2], 
                    color='blue', s=100, marker='x', label='Waypoints')
            
            # 시작점에서 첫 경로점까지 선 표시
            ax.plot([self.env.start_position[0], waypoints_array[0, 0]], 
                    [self.env.start_position[1], waypoints_array[0, 1]], 
                    [self.env.start_position[2], waypoints_array[0, 2]], 
                    'b--', alpha=0.7)
            
            # 경로점들 사이 선 표시
            for i in range(len(waypoints) - 1):
                ax.plot([waypoints_array[i, 0], waypoints_array[i+1, 0]], 
                        [waypoints_array[i, 1], waypoints_array[i+1, 1]], 
                        [waypoints_array[i, 2], waypoints_array[i+1, 2]], 
                        'b--', alpha=0.7)
            
            # 마지막 경로점에서 목표까지 선 표시
            ax.plot([waypoints_array[-1, 0], self.env.goal_position[0]], 
                    [waypoints_array[-1, 1], self.env.goal_position[1]], 
                    [waypoints_array[-1, 2], self.env.goal_position[2]], 
                    'b--', alpha=0.7)
            
            # 충돌 및 성공 시뮬레이션
            collision, goal_reached, path_length = self.env._simulate_path(waypoints)
            
            if collision:
                status_text.append("경로 충돌: O")
            else:
                status_text.append("경로 충돌: X")
                
            if goal_reached:
                status_text.append("목표 도달: O")
            else:
                status_text.append("목표 도달: X")
                
            status_text.append(f"경로 길이: {path_length:.3f}")
            status_text.append(f"경로점 개수: {len(waypoints)}")
            
            # 결과 출력
            print(f"  시뮬레이션 결과: {'성공' if goal_reached else '실패'} "
                f"{'(충돌)' if collision else ''}")
            print(f"  경로점 개수: {len(waypoints)}")
            print(f"  경로 길이: {path_length:.4f}")
        else:
            status_text.append("경로점: 없음")
        
        # 장애물 정보 추가
        status_text.append(f"장애물 개수: {len([o for o in self.env.obstacles if o['radius'] > 0])}")
        status_text.append(f"Iteration: {iteration}")
        
        # 그래프 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f'경로 계획 (Iteration {iteration})')
        
        # 상태 텍스트 표시
        status_str = '\n'.join(status_text)
        plt.figtext(0.02, 0.02, status_str, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # 범례 및 그리드
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # 다양한 각도 보기
        ax.view_init(elev=30, azim=45)
        
        # 저장
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # GIF용 다중 각도 저장
        if self.save_gif and waypoints:
            for i, angle in enumerate(range(0, 360, 30)):
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(111, projection='3d')
                
                # 환경 및 경로점 시각화 코드 (간결성을 위해 생략)
                # ... 
                
                # 각도만 변경
                ax.view_init(elev=30, azim=angle)
                
                # 저장
                angle_path = os.path.join(self.viz_dir, f"iteration_{iteration:04d}_angle_{i:02d}.png")
                plt.savefig(angle_path, dpi=100)
                plt.close(fig)
        
        return save_path

    def visualize_env_only(self, result, iteration):
        """환경 정보만 시각화 (경로점 없이)"""
        # 결과 저장 경로
        save_path = os.path.join(self.viz_dir, f"iteration_{iteration:04d}_env.png")
        
        # 결과 정보 추출
        reward_mean = result.get("episode_reward_mean", 0)
        reward_min = result.get("episode_reward_min", 0)
        reward_max = result.get("episode_reward_max", 0)
        
        # 3D 시각화
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 시작점 및 목표점 표시
        ax.scatter(self.env.start_position[0], self.env.start_position[1], self.env.start_position[2], 
                color='green', s=200, marker='o', label='Start')
        ax.scatter(self.env.goal_position[0], self.env.goal_position[1], self.env.goal_position[2], 
                color='red', s=200, marker='*', label='Goal')
        
        # 환경에 있는 장애물 시각화
        for i, obstacle in enumerate(self.env.obstacles):
            if obstacle["radius"] > 0:  # 유효한 장애물만
                # 장애물 위치 및 반지름
                pos = obstacle["position"]
                r = obstacle["radius"]
                
                # 실린더 바닥면 그리기 (원)
                theta = np.linspace(0, 2*np.pi, 50)
                x = pos[0] + r * np.cos(theta)
                y = pos[1] + r * np.sin(theta)
                z = np.zeros_like(theta)  # z=0 평면에 그리기
                
                # 실린더 윗면 그리기
                x2 = pos[0] + r * np.cos(theta)
                y2 = pos[1] + r * np.sin(theta)
                z2 = np.ones_like(theta)  # z=1 평면에 그리기
                
                # 실린더 표면 그리기
                ax.plot(x, y, z, 'gray', alpha=0.5)
                ax.plot(x2, y2, z2, 'gray', alpha=0.5)
                
                # 기둥선 그리기 (4개 모서리)
                for j in range(0, 50, 12):
                    ax.plot([x[j], x2[j]], [y[j], y2[j]], [z[j], z2[j]], 'gray', alpha=0.5, linewidth=0.7)
                
                # 장애물 ID 표시
                ax.text(pos[0], pos[1], pos[2] + 0.1, f"Obs{i+1}", color='black', fontsize=8)
                
                # 움직이는 장애물인 경우 방향 표시
                if obstacle["speed"] > 0:
                    dx = 0.2 * np.cos(obstacle["course"])
                    dy = 0.2 * np.sin(obstacle["course"])
                    ax.quiver(pos[0], pos[1], pos[2], dx, dy, 0, color='red', arrow_length_ratio=0.3)
        
        # 학습 진행 상태 텍스트 추가
        status_text = []
        status_text.append(f"Iteration: {iteration}")
        status_text.append(f"Avg Reward: {reward_mean:.2f}")
        status_text.append(f"Min Reward: {reward_min:.2f}")
        status_text.append(f"Max Reward: {reward_max:.2f}")
        status_text.append(f"장애물 개수: {len([o for o in self.env.obstacles if o['radius'] > 0])}")
        status_text.append("체크포인트를 찾을 수 없어 환경만 시각화합니다.")
        
        # 결과 출력
        print(f"  평균 보상: {reward_mean:.2f}")
        print(f"  최소 보상: {reward_min:.2f}")
        print(f"  최대 보상: {reward_max:.2f}")
        
        # 그래프 설정
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        ax.set_title(f'환경 상태 (Iteration {iteration})')
        
        # 상태 텍스트 표시
        status_str = '\n'.join(status_text)
        plt.figtext(0.02, 0.02, status_str, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # 범례 및 그리드
        ax.legend(loc='upper right')
        ax.grid(True)
        
        # 다양한 각도 보기
        ax.view_init(elev=30, azim=45)
        
        # 저장
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return save_path

    def plot_training_progress(self, result, iteration):
        """학습 진행 상황을 그래프로 시각화"""
        if not self.rewards_history:
            return
        
        # 그래프 저장 경로
        save_path = os.path.join(self.viz_dir, f"progress_{iteration:04d}.png")
        
        # 데이터 추출
        iterations = [x[0] for x in self.rewards_history]
        rewards = [x[1] for x in self.rewards_history]
        
        # 그래프 생성
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, rewards, 'b-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Average Reward')
        plt.title('Training Progress')
        plt.grid(True)
        
        # 최근 보상 표시
        if len(rewards) > 0:
            plt.text(0.02, 0.95, f"Current Reward: {rewards[-1]:.2f}", transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # 저장
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path

    def create_training_summary(self, final_result):
        """학습 완료 후 요약 정보 및 그래프 생성"""
        # 요약 파일 경로
        summary_path = os.path.join(self.result_dir, "training_summary.txt")
        
        # 주요 지표 추출
        metrics = {
            "총 학습 횟수": final_result.get("training_iteration", 0),
            "최종 평균 보상": final_result.get("episode_reward_mean", 0),
            "최종 최소 보상": final_result.get("episode_reward_min", 0),
            "최종 최대 보상": final_result.get("episode_reward_max", 0),
            "최종 평균 에피소드 길이": final_result.get("episode_len_mean", 0),
            "총 학습 시간(초)": final_result.get("time_total_s", 0),
        }
        
        # 요약 파일 작성
        with open(summary_path, 'w') as f:
            f.write("=== 학습 요약 ===\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
        print(f"학습 요약 저장 완료: {summary_path}")
        return summary_path


def create_visualizer_callback(env_creator, env_config, result_dir, viz_interval=10, max_viz=50, save_gif=False):
    """시각화 콜백 생성 헬퍼 함수"""
    return TrainingVisualizerCallback(
        env_creator=env_creator,
        env_config=env_config,
        result_dir=result_dir,
        visualization_interval=viz_interval,
        max_visualizations=max_viz,
        save_gif=save_gif
    )
