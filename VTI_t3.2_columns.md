# Data Columns Description for `data/VTI/t3.2`

This document describes the structure and columns of the CSV files located in `data/VTI/t3.2`.

## 1. Ego vehicle data
**Directory:** `Ego vehicle data`
**File Pattern:** `i4driving_road*_db_*.csv`
**Description:** High-frequency telemetry data from the ego vehicle, including physics (position, velocity, acceleration), control inputs (steering, pedals), eye-tracking data (OpenXR, Varjo), and scenario state.

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `time` | Simulation time in seconds. | `24.825` |
| `systemDatetime` | System timestamp. | `2024-03-18 13:36:38.268` |
| `dateTime` | Date and time. | `2024-03-18 13:36:38.038` |
| `oveOdometer` | Odometer reading (km). | `0.523` |
| `ovePositionRoad` | Road ID. | `176` |
| `ovePositionLongitudinalS` | Longitudinal position on road (S). | `89.477` |
| `ovePositionLateralR` | Lateral position on road (R). | `1.75` |
| `ovePositionVerticalH` | Vertical position (H). | `0` |
| `ovePositionYaw` | Yaw angle. | `3.134` |
| `ovePositionDrivingDirection` | Driving direction (-1 or 1). | `-1` |
| `oveInertialPositionX` | Global X position. | `1084.373` |
| `oveInertialPositionY` | Global Y position. | `-191.976` |
| `oveInertialPositionZ` | Global Z position. | `0` |
| `oveInertialHeadingPitchX` | Pitch angle. | `0` |
| `oveInertialHeadingRollY` | Roll angle. | `0` |
| `oveInertialHeadingYawZ` | Yaw angle (global). | `132.497` |
| `oveBodyVelocityX` | Body velocity X. | `1.75` |
| `oveBodyVelocityY` | Body velocity Y. | `0.007` |
| `oveBodyVelocityZ` | Body velocity Z. | `0` |
| `oveRoadVelocityLongitudinalS` | Road longitudinal velocity. | `1.75` |
| `oveRoadVelocityLateralR` | Road lateral velocity. | `-0.001` |
| `oveRoadVelocityVerticalH` | Road vertical velocity. | `0` |
| `oveBodyAccelerationLongitudinalX` | Body acceleration X. | `3.146` |
| `oveBodyAccelerationLateralY` | Body acceleration Y. | `0.02` |
| `oveBodyAccelerationVerticalZ` | Body acceleration Z. | `0` |
| `oveRoadAccelerationLongitudinalS` | Road acceleration S. | `3.117` |
| `oveRoadAccelerationLateralR` | Road acceleration R. | `-0.021` |
| `oveRoadAccelerationVerticalH` | Road acceleration H. | `0` |
| `oveBodyJerkLongitudinalX` | Body jerk X. | `0.039` |
| `oveBodyJerkLateralY` | Body jerk Y. | `0.027` |
| `oveBodyJerkVerticalZ` | Body jerk Z. | `0` |
| `oveRoadJerkLongitudinalS` | Road jerk S. | `0.043` |
| `oveRoadJerkLateralR` | Road jerk R. | `-0.196` |
| `oveRoadJerkVerticalH` | Road jerk H. | `0` |
| `oveYawVelocity` | Yaw velocity. | `0.005` |
| `engineRevolutionRad_per_sec` | Engine RPM (rad/s). | `100` |
| `engineTorque` | Engine torque. | `136.999` |
| `throttle` | Throttle pedal input (0-1). | `0.293` |
| `brakePedalActive` | Brake pedal active status. | `False` |
| `brakeForce` | Brake force applied. | `0.5` |
| `steeringWheelAngle` | Steering wheel angle. | `0.208` |
| `steeringWheelTorque` | Steering wheel torque. | `-0.033` |
| `indicatorLeft` | Left turn indicator. | `False` |
| `indicatorRight` | Right turn indicator. | `False` |
| `esminiScenario` | Current scenario name (if any). | `` |
| `esminiStatus` | Scenario status code. | `0` |
| `arrowsVisible` | Visibility of navigational arrows (NDRT?). | `False` |
| `arrowsCorrectCount` | Count of correct arrow responses. | `0` |
| `arrowsWrongCount` | Count of wrong arrow responses. | `0` |
| `arrowsTimeoutCount` | Count of arrow timeouts. | `0` |
| `aheadHeadway` | Headway to vehicle ahead. | `-1` |
| `aheadTHW` | Time headway to vehicle ahead. | `-1` |
| `aheadSpeed` | Speed of vehicle ahead. | `-1` |
| `workloadRating` | Workload rating input. | `0` |
| `workloadQuestionAsked` | Workload question status. | `0` |
| `bikeCrossingDistance` | Distance to crossing bike. | `0` |
| `pedestrianCrossingDistance` | Distance to crossing pedestrian. | `0` |
| `bikeBikerDistance` | Distance to biker. | `-1` |
| `oveInertialHeadingYawZFixed` | Fixed/Adjusted Yaw Z. | `132.497` |
| `openxrGazeHeading` | Eye gaze heading (OpenXR). | `-36.666` |
| `openxrGazePitch` | Eye gaze pitch (OpenXR). | `-8.788` |
| `openxrGazeWorldModel` | Object gaze is hitting. | `LeftMirror` |
| `openxrGazeWorldModelNumeric` | Numeric ID of object hit. | `6` |
| `bikeBikerDistanceRelative` | Relative distance to biker. | `NaN` |
| `varjoPupilDiameter` | Pupil diameter (Varjo). | `0` |
| `varjoPupilIrisRatio` | Pupil/Iris ratio (Varjo). | `0` |
| `varjoEyelidOpening` | Eyelid opening (Varjo). | `0.253` |
| `miraLeft` | Left mirror visibility/state. | `NaN` |
| `miraRight` | Right mirror visibility/state. | `NaN` |
| `miraForward` | Forward visibility/state. | `NaN` |
| `miraBehind` | Behind visibility/state. | `NaN` |

## 2. Event description
**Directory:** `Event description`
**Description:** metadata definitions for different scenario types.

### File: `bikecrossingscenarios.csv`
| Column Name | Description | Example |
| :--- | :--- | :--- |
| `Test case` | Identifier for the test case. | `Bike1` |
| `Turning manoeuvre` | Ego vehicle maneuver. | `Right` |
| `Bicylist present` | Whether a cyclist is present. | `Yes` |
| `Traffic from the sides` | Description of side traffic. | `-` |

### File: `carfollowingscenarios.csv`
| Column Name | Description | Example |
| :--- | :--- | :--- |
| `Test case` | Identifier for the test case. | `CF1` |
| `Priority rule intersection 1` | Rule at first intersection. | `Main` |
| `Lead vehicles` | Number of lead vehicles. | `1` |
| `Follower` | Follower vehicle presence. | `1` |
| `Priority rule intersection 2` | Rule at second intersection. | `Yield` |
| `Clear traffic` | Condition for clearing traffic. | `Lead1 and follower right` |

### File: `pedestrianscenarios.csv`
| Column Name | Description | Example |
| :--- | :--- | :--- |
| `Test case` | Identifier for the test case. | `PedestrianCrossingP1` |
| `Pedestrian to the Left` | Behavior of pedestrian on left. | `-` |
| `Pedestrian to the Right` | Behavior of pedestrian on right. | `-` |

## 3. NDRT (Non-Driving Related Task)
**Directory:** `NDRT`
**File Pattern:** `i4driving_road*_db_*.csv`
**Description:** Logs of user responses to a secondary task (likely the "arrows" mentioned in Ego data).

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `datetime` | Timestamp of the event. | `2024-03-18 13:38:31.031` |
| `type` | Type of event/result. | `correct` |
| `ix` | Index or ID of the event. | `5682` |
| `episode` | Episode number. | `1` |
| `responseTime` | Time taken to respond (seconds). | `2.701` |

## 4. Participant information
**Directory:** `Participant information`
**File:** `questionnaire.csv`
**Description:** Demographics, driving experience, and Driver Style Inventory (DSI) scores for participants.

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `participant` | Participant ID. | `2` |
| `exclude` | Exclusion criteria flag. | `No` |
| `age` | Participant age. | `44` |
| `gender` | Participant gender. | `Man` |
| `experienceDrivingLicenseYears` | Years with license. | `27` |
| `experienceKmDrivenPastYear` | Kilometers driven last year. | `40000` |
| `experienceUrbanDriving` | Frequency of urban driving. | `1 Daily` |
| `comfortableUrbanDriving` | Comfort level in urban driving. | `5 Very comfortable` |
| ... | ... | ... |
| `DSI*_...` | Various Driver Style Inventory scores (1-20+ columns). | `5` |
| `sim_...` | Simulator sickness and realism ratings. | `4` |
| `DSI_SafetyScore` | Aggregated safety score. | `4.6` |

*(Note: Table truncated for brevity due to large number of questionnaire columns)*

## 5. Physiology
**Directory:** `Physiology`
**File Pattern:** `i4driving_road*_db_*.csv`
**Description:** Physiological sensor data (ECG).

| Column Name | Description | Example |
| :--- | :--- | :--- |
| `time` | Timestamp. | `24.82859375` |
| `ecg` | ECG signal value. | `-0.0393263493882086` |
| `beats` | Detected heart beats (binary/flag). | `0` |

## 6. Surrounding vehicle data
**Directory:** `Surrounding vehicle data`
**File Pattern:** `i4driving_road*_scene_*.csv`
**Description:** A semi-colon separated file describing the state of all actors (vehicles, pedestrians, etc.) in the scene at each timestamp.
**Format:** Non-standard tabular format. Columns are dynamic based on the number of actors.

**Structure:**
`timer;system_timestamp;scene_count;[Actor1_Data];[Actor2_Data];...`

**Sample Block:**
`24.855;240318 133638.298;11;262147;speed_limit_40;Road 176;...`

**Actor Data Fields (Inferred):**
The repeating blocks contain data like:
- ID / Type ID (`262147`)
- Name (`speed_limit_40`, `kia_ceed_blue`)
- Road (`Road 176`)
- Position coordinates (X, Y, Z, etc.)
- Orientation/Heading
