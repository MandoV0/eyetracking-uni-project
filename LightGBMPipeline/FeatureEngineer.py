import pandas as pd
import numpy as np

class FeatureEngineer:    
    def engineer_features(self, df, window_size=50):
        """
        Engineer features from raw sensor data using rolling windows
        
        Args:
            df: Raw data DataFrame
            window_size: Number of rows for rolling calculations
        """
        print("Engineering features...")
        features = pd.DataFrame()
        
        features['time'] = df['time']
        
        # AGGRESSIVE DRIVING INDICATORS
        # Acceleration metrics
        features['accel_long_mean'] = df['oveBodyAccelerationLongitudinalX'].rolling(window_size, min_periods=1).mean()
        features['accel_long_std'] = df['oveBodyAccelerationLongitudinalX'].rolling(window_size, min_periods=1).std()
        features['accel_long_max'] = df['oveBodyAccelerationLongitudinalX'].rolling(window_size, min_periods=1).max()
        features['accel_lat_std'] = df['oveBodyAccelerationLateralY'].rolling(window_size, min_periods=1).std()
        
        # Jerk metrics. The rate of change in acceleration
        features['jerk_long_mean'] = df['oveBodyJerkLongitudinalX'].rolling(window_size, min_periods=1).mean().abs()
        features['jerk_long_max'] = df['oveBodyJerkLongitudinalX'].rolling(window_size, min_periods=1).max().abs()
        features['jerk_lat_std'] = df['oveBodyJerkLateralY'].rolling(window_size, min_periods=1).std()
        
        # Speed and throttle
        features['speed'] = np.sqrt(df['oveBodyVelocityX'] ** 2 + df['oveBodyVelocityY']**2)
        features['speed_std'] = features['speed'].rolling(window_size, min_periods=1).std()
        features['throttle_mean'] = df['throttle'].rolling(window_size, min_periods=1).mean()
        features['throttle_changes'] = df['throttle'].diff().abs().rolling(window_size, min_periods=1).sum()
        
        # Steering aggressiveness
        features['steering_angle_std'] = df['steeringWheelAngle'].rolling(window_size, min_periods=1).std()
        features['steering_rate'] = df['steeringWheelAngle'].diff().abs().rolling(window_size, min_periods=1).mean()
        features['yaw_velocity_std'] = df['oveYawVelocity'].rolling(window_size, min_periods=1).std()
        
        # Headway is the distance to car in front of us
        if 'aheadTHW' in df.columns:
            features['thw_mean'] = df['aheadTHW'].replace(-1, np.nan).rolling(window_size, min_periods=1).mean()
            features['short_headway_ratio'] = (df['aheadTHW'].rolling(window_size, min_periods=1).apply(
                lambda x: (((x > 0) & (x < 2)).sum() / len(x)) if len(x) > 0 else 0
            ))
        
        # Brake usage
        if 'brakePedalActive' in df.columns:
            features['brake_frequency'] = df['brakePedalActive'].rolling(window_size, min_periods=1).sum()
        
        # Surrounding vehicle features
        if 'surround_min_dist' in df.columns:
            features['surround_min_dist'] = df['surround_min_dist'].rolling(window_size, min_periods=1).mean()
            features['surround_actor_count'] = df['surround_actor_count'].rolling(window_size, min_periods=1).mean()
        
        # INATTENTIVE DRIVING INDICATORS
        # Lane position variability. How much the driver is drifting from his lane.
        # Happens if the driver is not paying attention to the road, because he is looking at his phone or something.
        features['lateral_pos_std'] = df['ovePositionLateralR'].rolling(window_size, min_periods=1).std()
        features['lateral_pos_mean'] = df['ovePositionLateralR'].rolling(window_size, min_periods=1).mean().abs()
        
        # NDRT (Non driving related tasks) performance.
        if 'arrowsWrongCount' in df.columns:
            features['ndrt_error_rate'] = (df['arrowsWrongCount'] + df['arrowsTimeoutCount']).rolling(window_size, min_periods=1).mean()
            features['ndrt_total_attempts'] = (df['arrowsCorrectCount'] + df['arrowsWrongCount'] + df['arrowsTimeoutCount']).rolling(window_size, min_periods=1).sum()
        
        # EYE GAZE FEATURES
        if 'openxrGazeHeading' in df.columns:
            features['gaze_heading_abs'] = df['openxrGazeHeading'].abs()
            features['gaze_pitch_abs'] = df['openxrGazePitch'].abs()
            features['gaze_off_road'] = ((df['openxrGazeHeading'].abs() > 30) | 
                                          (df['openxrGazePitch'].abs() > 20)).astype(int)
            features['gaze_off_road_ratio'] = features['gaze_off_road'].rolling(window_size, min_periods=1).mean()
        
        # Eyelid opening. Used for drowsiness detection
        # 1 is fully open, 0 is fully closed.
        if 'varjoEyelidOpening' in df.columns:
            features['eyelid_mean'] = df['varjoEyelidOpening'].rolling(window_size, min_periods=1).mean()
            features['eyelid_low_ratio'] = (df['varjoEyelidOpening'] < 0.3).rolling(window_size, min_periods=1).mean()
        
        # Control smoothness. The smoother you drive the more attentive the driver.
        # Smoothing based on how much we steer and how much we accelerate/break
        features['control_smoothness'] = 1 / (1 + features['steering_rate'] + features['throttle_changes'] / 10)
        
        # Physiological features
        if 'heartRate' in df.columns:
            features['hr_mean'] = df['heartRate'].rolling(window_size, min_periods=1).mean()
            features['hr_change'] = df['heartRate'].diff().abs().rolling(window_size, min_periods=1).mean()
        
        if 'rrInterval' in df.columns:
            # HRV (Heart Rate Variability) - SDNN
            features['hrv_sdnn'] = df['rrInterval'].rolling(window_size, min_periods=1).std()
            # RMSSD (Root Mean Square of Successive Differences)
            diff_rr = df['rrInterval'].diff()
            features['hrv_rmssd'] = (diff_rr ** 2).rolling(window_size, min_periods=1).mean() ** 0.5
        
        # Fill NaN values
        features = features.bfill().ffill().fillna(0)
        
        print(f"Engineered {len(features.columns)} features from {len(df)} samples")
        return features

