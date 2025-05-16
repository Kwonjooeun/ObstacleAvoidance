import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
from typing import Dict, List, Tuple, Optional

class ObstacleAvoidanceEnv(gym.Env):
    def __init__(self, env_config):
        super(ObstacleAvoidanceEnv, self).__init__()

        # 환경 설정 파라미터
        self.max_obstacles = env_config.get("max_obstacles", 16)
        self.current_max_obstacles = env_config.get("current_max_obstacles", 1)  # 커리큘럼 러닝용
        self.max_steps = env_config.get("max_steps", 10)  # 경로점 기반이므로 에피소드 스텝 수 감소
        self.step_size = env_config.get("step_size", 0.05)  # 충돌 검사용 세분화 스텝
        self.collision_threshold = env_config.get("collision_threshold", 0.1)
        self.goal_threshold = env_config.get("goal_threshold", 0.1)
        self.obstacle_speed_range = env_config.get("obstacle_speed_range", (0.0, 0.05))
        self.max_waypoints = env_config.get("max_waypoints", 8)  # 최대 경로점 개수

        # 장애물 개수 분포 (지정되지 않으면 current_max_obstacles 사용)
        self.obstacle_distribution = env_config.get("obstacle_distribution", None)
        
        # 시작점과 목표점 설정
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.goal_position = np.array([1.0, 1.0, 1.0])
        
        # 현재 위치 초기화
        self.current_position = self.start_position.copy()
        self.steps_taken = 0
        self.waypoints = []  # 현재 에피소드의 경로점 저장
        
        # Action space: 경로점 개수(1) + 경로점 좌표(8개 * 3좌표) + 각 경로점 활성화 여부(8개)
        # [num_waypoints, wp1_x, wp1_y, wp1_z, wp1_active, wp2_x, ... wp8_active]
        self.action_space = spaces.Box(
            low=np.array([1.0] + [-1.0, -1.0, -1.0, 0.0] * self.max_waypoints),
            high=np.array([float(self.max_waypoints)] + [1.0, 1.0, 1.0, 1.0] * self.max_waypoints),
            dtype=np.float32
        )

        # Observation space:
        # [
        #   무인기의 현재 위치(3), 
        #   목표지점까지의 차이(3),
        #   각 장애물 정보(16개): [x, y, z, 반지름, 속력, 진행방향(코스)]
        # ]
        obs_dim = 3 + 3 + self.max_obstacles * 6
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(obs_dim,), dtype=np.float32)

        # 장애물 정보
        self.obstacles = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # 현재 위치 초기화
        self.current_position = self.start_position.copy()
        self.steps_taken = 0
        self.waypoints = []
        
        # 장애물 생성
        self.obstacles = self._generate_obstacles()
        
        # 초기 관측 정보 반환
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        self.steps_taken += 1
        
        # 액션에서 경로점 정보 추출
        waypoints = self._get_waypoints_from_action(action)
        self.waypoints = waypoints  # 현재 경로점 저장
        
        # 경로점 개수 제한 확인 (1~8개)
        num_waypoints = len(waypoints)
        if num_waypoints < 1 or num_waypoints > self.max_waypoints:
            return self._get_observation(), -10.0, False, True, self._get_info()  # 잘못된 경로점 개수
        
        # 장애물 위치 업데이트 (동적 장애물)
        self._update_obstacles()
        
        # 경로점을 따라 시뮬레이션 진행
        collision, goal_reached, path_length = self._simulate_path(waypoints)
        
        # 보상 및 종료 조건 계산
        reward, terminated, truncated = self._compute_reward(waypoints, collision, goal_reached, path_length)
        
        # 최대 스텝 수 초과 체크
        if self.steps_taken >= self.max_steps:
            truncated = True
        
        # 관측 정보 반환
        observation = self._get_observation()
        info = self._get_info(collision, goal_reached, path_length)
        
        return observation, reward, terminated, truncated, info

    def _get_waypoints_from_action(self, action):
        """액션에서 유효한 경로점 추출"""
        waypoints = []
        
        # 첫 번째 값은 경로점 개수
        num_waypoints = int(max(1, min(self.max_waypoints, round(action[0]))))
        
        # 경로점 추출 (4개 값씩 하나의 경로점 정보)
        for i in range(num_waypoints):
            idx = 1 + i * 4  # 인덱스 0은 경로점 개수이므로 1부터 시작
            x, y, z, active = action[idx:idx+4]
            
            # 활성화 값이 0.5 이상이면 유효한 경로점으로 간주
            if active >= 0.5:
                # 좌표값을 0~1 범위로 정규화 (-1~1 → 0~1)
                scaled_pos = np.array([(x + 1) / 2, (y + 1) / 2, (z + 1) / 2])
                waypoints.append(scaled_pos)
        
        # 유효한 경로점이 없으면 목표 지점을 유일한 경로점으로 설정
        if len(waypoints) == 0:
            waypoints.append(np.array([0.5, 0.5, 0.5]))  # 환경 중앙에 하나의 경로점 생성
        
        return waypoints

    def _simulate_path(self, waypoints):
        """경로점을 따라 가상으로 시뮬레이션하여 충돌 여부와 경로 길이 계산"""
        current_pos = self.start_position.copy()
        path_length = 0.0
        collision = False
        
        # 각 경로점을 순서대로 방문
        for waypoint in waypoints:
            # 현재 위치에서 경로점까지의 직선 경로 시뮬레이션
            direction = waypoint - current_pos
            distance = np.linalg.norm(direction)
            path_length += distance
            
            if distance > 0:
                # 경로를 작은 스텝으로 나누어 충돌 검사
                num_steps = max(1, int(distance / self.step_size))
                step_vector = direction / num_steps
                
                for _ in range(num_steps):
                    current_pos += step_vector
                    if self._check_collision(current_pos):
                        collision = True
                        break
                
                if collision:
                    break
        
        # 마지막 경로점에서 목표 지점까지의 경로 시뮬레이션
        goal_reached = False
        if not collision:
            direction = self.goal_position - current_pos
            distance = np.linalg.norm(direction)
            path_length += distance
            
            if distance > 0:
                num_steps = max(1, int(distance / self.step_size))
                step_vector = direction / num_steps
                
                for _ in range(num_steps):
                    current_pos += step_vector
                    if self._check_collision(current_pos):
                        collision = True
                        break
                    
                    # 목표 지점 도달 확인
                    if np.linalg.norm(current_pos - self.goal_position) < self.goal_threshold:
                        goal_reached = True
                        break
            else:
                goal_reached = True
        
        return collision, goal_reached, path_length

    def _generate_obstacles(self):
        """무작위 장애물 생성 - 분포 지원"""
        obstacles = []
        
        # 장애물 개수 결정
        if self.obstacle_distribution:
            # 분포에서 장애물 개수 샘플링
            num_obstacles = np.random.choice(
                list(self.obstacle_distribution.keys()),
                p=list(self.obstacle_distribution.values())
            )
        else:
            # 기존 방식: 1에서 현재 최대까지 균등 분포
            num_obstacles = np.random.randint(0, self.current_max_obstacles + 1)
        
        for _ in range(num_obstacles):
            # 장애물 위치는 0과 1 사이에서 무작위로 생성
            x = np.random.uniform(0.2, 0.8)
            y = np.random.uniform(0.2, 0.8)
            z = np.random.uniform(0.2, 0.8)
            
            # 반지름은 0.05에서 0.15 사이로 설정
            radius = np.random.uniform(0.05, 0.15)
            
            # 속력 설정 (0이면 정지, 양수면 동적)
            speed = np.random.choice([0, np.random.uniform(*self.obstacle_speed_range)])
            
            # 진행방향(코스) 설정 (라디안)
            course = np.random.uniform(0, 2*np.pi) if speed > 0 else 0
            
            obstacles.append({
                "position": np.array([x, y, z]),
                "radius": radius,
                "speed": speed,
                "course": course
            })
        
        # 남은 장애물 슬롯을 제로 패딩으로 채우기
        for _ in range(self.max_obstacles - num_obstacles):
            obstacles.append({
                "position": np.zeros(3),
                "radius": 0,
                "speed": 0,
                "course": 0
            })
        
        return obstacles

    def _update_obstacles(self):
        """동적 장애물 위치 업데이트"""
        for obstacle in self.obstacles:
            if obstacle["speed"] > 0:  # 움직이는 장애물만 업데이트
                # 장애물의 이동 방향 계산
                dx = obstacle["speed"] * math.cos(obstacle["course"])
                dy = obstacle["speed"] * math.sin(obstacle["course"])
                dz = 0  # 2D 평면에서만 이동
                
                # 장애물 위치 업데이트
                obstacle["position"][0] += dx
                obstacle["position"][1] += dy
                obstacle["position"][2] += dz
                
                # 경계 체크 (0~1 범위)
                for i in range(3):
                    if obstacle["position"][i] < 0 or obstacle["position"][i] > 1:
                        # 방향 반전
                        obstacle["course"] = (obstacle["course"] + math.pi) % (2 * math.pi)
                        obstacle["position"][i] = np.clip(obstacle["position"][i], 0, 1)
                        break

    def _check_collision(self, position):
        """장애물과의 충돌 여부 확인"""
        for obstacle in self.obstacles:
            if obstacle["radius"] > 0:  # 유효한 장애물인 경우
                # 3D 거리 계산 (x,y만 고려하는 실린더 형태)
                obstacle_pos = obstacle["position"]
                dx = position[0] - obstacle_pos[0]
                dy = position[1] - obstacle_pos[1]
                dist_2d = np.sqrt(dx**2 + dy**2)
                
                # 장애물 반지름보다 거리가 작으면 충돌
                if dist_2d < obstacle["radius"] + self.collision_threshold:
                    return True
        
        return False        

    def _compute_reward(self, waypoints, collision, goal_reached, path_length):
        """보상 및 종료 상태 계산"""
        # 기본 보상 설정
        reward = 0.0
        
        # 충돌 패널티
        if collision:
            reward -= 10.0
            return reward, False, True  # 종료 (실패)
        
        # 목표 도달 보상
        if goal_reached:
            # 직선 거리 vs 실제 경로 비율로 효율성 계산
            direct_distance = np.linalg.norm(self.goal_position - self.start_position)
            efficiency = direct_distance / max(path_length, direct_distance)
            
            # 기본 성공 보상 + 효율성 보너스 + 경로점 개수에 따른 보너스
            base_reward = 10.0
            efficiency_bonus = 10.0 * efficiency  # 효율적일수록 보상 증가
            waypoint_bonus = 5.0 * (1.0 - (len(waypoints) / self.max_waypoints))  # 적은 경로점 사용 시 보상
            
            reward = base_reward + efficiency_bonus + waypoint_bonus
            return reward, True, False  # 종료 (성공)
        
        # 진행 중인 경우 (충돌도 없고 목표 도달도 안 됨)
        # 경로점 개수에 따른 작은 패널티로 경로점 최소화 유도
        waypoint_penalty = -0.1 * len(waypoints)
        reward += waypoint_penalty
        
        return reward, False, False  # 계속 진행

    def _get_observation(self):
        """현재 상태에 대한 관측 데이터 생성"""
        # 현재 위치
        observation = np.copy(self.current_position)
        
        # 목표점까지의 상대적 위치 (E, N, U 차이)
        goal_relative = self.goal_position - self.current_position
        observation = np.concatenate([observation, goal_relative])
        
        # 장애물 정보
        for obstacle in self.obstacles:
            # 장애물 정보: 위치(3), 반지름(1), 속력(1), 코스(1)
            obstacle_info = np.concatenate([
                obstacle["position"],
                [obstacle["radius"], obstacle["speed"], obstacle["course"]]
            ])
            observation = np.concatenate([observation, obstacle_info])
        
        return observation
    
    def _get_info(self, collision=False, goal_reached=False, path_length=0.0):
        """추가 정보(메타데이터) 반환"""
        return {
            "num_obstacles": len([o for o in self.obstacles if o["radius"] > 0]),
            "distance_to_goal": np.linalg.norm(self.current_position - self.goal_position),
            "num_waypoints": len(self.waypoints),
            "path_length": path_length,
            "collision": collision,
            "success": goal_reached
        }
