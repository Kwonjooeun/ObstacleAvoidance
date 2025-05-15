import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from obstacle_avoidance import ObstacleAvoidanceEnv

def plot_cylinder(ax, x, y, z, radius, height, color='gray', alpha=0.3):
    """3D 원통형 장애물 그리기"""
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = x + radius * np.cos(theta)
    y_circle = y + radius * np.sin(theta)
    z_bottom = z - height/2 * np.ones_like(theta)
    z_top = z + height/2 * np.ones_like(theta)
    
    # 원통의 옆면
    for i in range(len(theta)-1):
        ax.plot([x_circle[i], x_circle[i+1]], 
                [y_circle[i], y_circle[i+1]], 
                [z_bottom[i], z_bottom[i+1]], 'k-', alpha=0.3)
        ax.plot([x_circle[i], x_circle[i+1]], 
                [y_circle[i], y_circle[i+1]], 
                [z_top[i], z_top[i+1]], 'k-', alpha=0.3)
    
    # 옆면 연결선
    for i in range(0, len(theta), 10):
        ax.plot([x_circle[i], x_circle[i]], 
                [y_circle[i], y_circle[i]], 
                [z_bottom[i], z_top[i]], 'k-', alpha=0.3)
    
    # 표면 색상
    ax.plot_surface = lambda x, y, z, **kwargs: None  # 표면 그리기 비활성화
    
    # 원 그리기
    ax.plot(x_circle, y_circle, z_bottom, color='black', alpha=0.5)
    ax.plot(x_circle, y_circle, z_top, color='black', alpha=0.5)

def visualize_trajectory(env, trajectory, output_dir, iteration, episode=0):
    """Agent의 경로와 장애물을 3D로 시각화"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 시작점과 목표점 표시
    ax.scatter([env.start_position[0]], [env.start_position[1]], [env.start_position[2]], 
               color='blue', marker='o', s=100, label='시작점')
    ax.scatter([env.goal_position[0]], [env.goal_position[1]], [env.goal_position[2]], 
               color='green', marker='*', s=200, label='목표점')
    
    # 경로 표시
    positions = np.array(trajectory)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=1, alpha=0.7)
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='purple', s=15, alpha=0.6, label='경로점')
    
    # 장애물 표시
    for obstacle in env.obstacles:
        if obstacle["radius"] > 0:  # 유효한 장애물만 표시
            pos = obstacle["position"]
            radius = obstacle["radius"]
            # 원통형 장애물 그리기
            plot_cylinder(ax, pos[0], pos[1], pos[2], radius, 2.0)
    
    # 축 범위 설정
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    
    # 축 레이블
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'강화학습 Agent 경로 시각화 (Iteration {iteration}, Episode {episode})')
    ax.legend()
    
    # 그래프 저장
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/trajectory_iter_{iteration}_ep_{episode}.png", dpi=200)
    plt.close()
# Visualization_Callback.py 수정
class VisualizationCallback(DefaultCallbacks):
    """단계별 디버깅을 위한 콜백"""
    
    def on_train_result(self, *, algorithm, result, **kwargs):
        """훈련 결과가 생성될 때마다 호출되는 콜백"""
        try:
            iteration = result["training_iteration"]
            print(f"\n=== 콜백 호출됨: Iteration {iteration} ===")
            
            # 10 iteration마다 진행
            if iteration % 10 == 0:
                print(f"Iteration {iteration}에서 시각화 시도 중...")
                
                # 1단계: 간단한 정보 출력
                print("평균 보상:", result.get("episode_reward_mean", "알 수 없음"))
                print("에피소드 수:", result.get("episodes_total", "알 수 없음"))
                
                # 2단계: 환경 설정 접근 시도
                try:
                    # 여러 경로로 환경 설정 접근 시도
                    env_config = None
                    
                    # 방법 1
                    if hasattr(algorithm, "config") and "env_config" in algorithm.config:
                        env_config = algorithm.config["env_config"]
                        print("방법 1로 환경 설정 접근 성공")
                    # 방법 2
                    elif hasattr(algorithm, "get_config") and callable(algorithm.get_config):
                        config = algorithm.get_config()
                        if "env_config" in config:
                            env_config = config["env_config"]
                            print("방법 2로 환경 설정 접근 성공")
                    
                    if env_config:
                        print("환경 설정 접근 성공:", list(env_config.keys()))
                    else:
                        print("환경 설정 접근 실패, 기본 설정 사용")
                        env_config = {
                            "max_obstacles": 16,
                            "current_max_obstacles": 16,
                            "max_steps": 100,
                            "step_size": 0.05,
                            "collision_threshold": 0.05,
                            "goal_threshold": 0.1,
                            "obstacle_speed_range": (0.0, 0.03)
                        }
                    
                    # 3단계: 환경 생성 시도
                    print("환경 생성 시도...")
                    from obstacle_avoidance import ObstacleAvoidanceEnv
                    env = ObstacleAvoidanceEnv(env_config)
                    print("환경 생성 성공")
                    
                    # 4단계: 간단한 시각화 디렉토리 설정
                    output_dir = os.path.join("results", "visualization", f"iter_{iteration}")
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"출력 디렉토리 생성: {output_dir}")
                    
                    # 5단계: 1개 에피소드 실행 및 경로 수집
                    print("에피소드 실행 중...")
                    obs, info = env.reset()
                    trajectory = [env.current_position.copy()]
                    
                    done = False
                    truncated = False
                    steps = 0
                    max_steps = 100
                    
                    while not (done or truncated) and steps < max_steps:
                        # 정책에서 액션 가져오기
                        action = algorithm.compute_single_action(obs, explore=False)
                        # 환경에 액션 적용
                        obs, reward, done, truncated, info = env.step(action)
                        # 경로 저장
                        trajectory.append(env.current_position.copy())
                        steps += 1
                    
                    print(f"에피소드 실행 완료: {steps}단계, 성공: {info.get('success', False)}")
                    
                    # 6단계: 간단한 2D 경로 시각화 (먼저 이것이 작동하는지 확인)
                    print("2D 경로 시각화 중...")
                    plt.figure(figsize=(8, 8))
                    
                    # 시작점과 목표점
                    plt.scatter(env.start_position[0], env.start_position[1], c='blue', marker='o', s=100, label='시작점')
                    plt.scatter(env.goal_position[0], env.goal_position[1], c='green', marker='*', s=200, label='목표점')
                    
                    # 경로 그리기
                    positions = np.array(trajectory)
                    plt.plot(positions[:, 0], positions[:, 1], 'r-', linewidth=1)
                    plt.scatter(positions[:, 0], positions[:, 1], c='purple', s=15, alpha=0.6)
                    
                    # 장애물 그리기
                    for obstacle in env.obstacles:
                        if obstacle["radius"] > 0:
                            pos = obstacle["position"]
                            circle = plt.Circle((pos[0], pos[1]), obstacle["radius"], color='gray', alpha=0.3)
                            plt.gca().add_patch(circle)
                    
                    plt.xlim(-0.1, 1.1)
                    plt.ylim(-0.1, 1.1)
                    plt.title(f'2D 경로 시각화 (Iteration {iteration})')
                    plt.legend()
                    plt.savefig(f"{output_dir}/trajectory_2d.png")
                    plt.close()
                    print(f"2D 경로 시각화 완료: {output_dir}/trajectory_2d.png")
                    
                    # 7단계: 3D 경로 시각화 시도 (2D가 성공한 후)
                    print("3D 경로 시각화 시도...")
                    self.visualize_3d_trajectory(env, trajectory, output_dir, iteration)
                    print(f"3D 경로 시각화 완료: {output_dir}/trajectory_3d.png")
                    
                except Exception as e:
                    import traceback
                    print(f"시각화 과정 중 오류 발생: {e}")
                    print(traceback.format_exc())
        
        except Exception as e:
            import traceback
            print(f"콜백 최상위 오류: {e}")
            print(traceback.format_exc())
    
    def visualize_3d_trajectory(self, env, trajectory, output_dir, iteration):
        """3D 경로 시각화 - 별도 메서드로 분리"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 시작점과 목표점
        ax.scatter([env.start_position[0]], [env.start_position[1]], [env.start_position[2]], 
                   color='blue', marker='o', s=100, label='시작점')
        ax.scatter([env.goal_position[0]], [env.goal_position[1]], [env.goal_position[2]], 
                   color='green', marker='*', s=200, label='목표점')
        
        # 경로 그리기
        positions = np.array(trajectory)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=1)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='purple', s=15, alpha=0.6)
        
        # 장애물 그리기 (간단한 형태로)
        for obstacle in env.obstacles:
            if obstacle["radius"] > 0:
                # 장애물 위치
                pos = obstacle["position"]
                
                # 원통 상단/하단 원 그리기
                theta = np.linspace(0, 2*np.pi, 20)
                x_circle = pos[0] + obstacle["radius"] * np.cos(theta)
                y_circle = pos[1] + obstacle["radius"] * np.sin(theta)
                z_bottom = pos[2] - 0.5  # 하단
                z_top = pos[2] + 0.5     # 상단
                
                # 원 그리기
                ax.plot(x_circle, y_circle, [z_bottom]*len(theta), 'k-', alpha=0.3)
                ax.plot(x_circle, y_circle, [z_top]*len(theta), 'k-', alpha=0.3)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_zlim(-0.1, 1.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D 경로 시각화 (Iteration {iteration})')
        ax.legend()
        
        plt.savefig(f"{output_dir}/trajectory_3d.png")
        plt.close()
