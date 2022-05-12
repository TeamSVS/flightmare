
#include "flightlib/envs/vision_env/vision_env.hpp"

namespace flightlib {

VisionEnv::VisionEnv()
  : VisionEnv(getenv("FLIGHTMARE_PATH") +
                std::string("/flightpy/configs/vision/config.yaml"),
              0) {}

VisionEnv::VisionEnv(const std::string &cfg_path, const int env_id)
  : EnvBase() {
  // check if configuration file exist
  if (!(file_exists(cfg_path))) {
    logger_.error("Configuration file %s does not exists.", cfg_path);
  }
  // load configuration file
  cfg_ = YAML::LoadFile(cfg_path);
  //
  init();
  env_id_ = env_id;
}

VisionEnv::VisionEnv(const YAML::Node &cfg_node, const int env_id) : EnvBase() {
  cfg_ = cfg_node;

  //
  init();
  env_id_ = env_id;
}

void VisionEnv::init() {
  //
  is_collision_ = false;
  unity_render_offset_ << 0.0, 0.0, 0.0;
  goal_linear_vel_ << 0.0, 0.0, 0.0;
  cmd_.setZeros();

  // create quadrotors
  quad_ptr_ = std::make_shared<Quadrotor>();
  quadrotorScale_ = 1.0;

  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quad_ptr_->updateDynamics(dynamics);


  // define input and output dimension for the environment
  obs_dim_ = visionenv::kNObs;
  act_dim_ = visionenv::kNAct;
  rew_dim_ = 0;
  num_detected_obstacles_ = visionenv::kNObstacles;

  // load parameters
  loadParam(cfg_);

  // add camera
  if (!configCamera(cfg_)) {
    logger_.error(
      "Cannot config RGB Camera. Something wrong with the config file");
  }

  obstacle_cfg_path_ = getenv("FLIGHTMARE_PATH") +
                       std::string("/flightpy/configs/vision/") +
                       difficulty_level_ + std::string("/") + env_folder_;

  // add dynamic objects
  std::string dynamic_object_yaml =
    obstacle_cfg_path_ + std::string("/dynamic_obstacles.yaml");
  if (!configDynamicObjects(dynamic_object_yaml)) {
    logger_.error(
      "Cannot config Dynamic Object Yaml. Something wrong with the config "
      "file");
  }

  // add static objects
  static_object_csv_ =
    obstacle_cfg_path_ + std::string("/static_obstacles.csv");
  if (!configStaticObjects(static_object_csv_)) {
    logger_.error(
      "Cannot config Static Object. Something wrong with the config file");
  }

  // use single rotor control or bodyrate control
  Scalar max_force = quad_ptr_->getDynamics().getForceMax();
  Vector<3> max_omega = quad_ptr_->getDynamics().getOmegaMax();
  //
  act_mean_ << (max_force / quad_ptr_->getMass()) / 2, 0.0, 0.0, 0.0;
  act_std_ << (max_force / quad_ptr_->getMass()) / 2, max_omega.x(),
    max_omega.y(), max_omega.z();
}

VisionEnv::~VisionEnv() {}

bool VisionEnv::reset(Ref<Vector<>> obs) {
  quad_state_.setZero();
  pi_act_.setZero();
  old_pi_act_.setZero();
  is_collision_ = false;

  // randomly reset the quadrotor state
  // reset position
  quad_state_.x(QS::POSX) = uniform_dist_(random_gen_);
  quad_state_.x(QS::POSY) = uniform_dist_(random_gen_) * 9.0;
  quad_state_.x(QS::POSZ) = uniform_dist_(random_gen_) * 4 + 5.0;

  max_dist_ = goal_pos_ - quad_state_.p;
  num_collision = 0;
  xMax = 0;

//  std::cout << "Reset!\n";
//  std::cout << "Starting Drone X:" << quad_state_.p(QS::POSX) << "\n";
//  std::cout << "Starting Drone Y:" << quad_state_.p(QS::POSY) << "\n";
//  std::cout << "Starting Drone Z:" << quad_state_.p(QS::POSZ) << "\n";

  // reset quadrotor with random states
  quad_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  // use collective thrust and bodyrate control mode
  cmd_.setCmdMode(quadcmd::THRUSTRATE);
  cmd_.collective_thrust = 0;
  cmd_.omega.setZero();

  // obtain observations
  getObs(obs);
  return true;
}

bool VisionEnv::reset(Ref<Vector<>> obs, bool random) { return reset(obs); }

bool VisionEnv::getObs(Ref<Vector<>> obs) {
  if (obs.size() != obs_dim_) {
    logger_.error("Observation dimension mismatch. %d != %d", obs.size(),
                  obs_dim_);
    return false;
  }
  // compute rotation matrix
  Vector<9> ori = Map<Vector<>>(quad_state_.R().data(), quad_state_.R().size());

  // get N most closest obstacles as the observation
  Vector<visionenv::kNObstacles * visionenv::kNObstaclesState> obstacle_obs;
  getObstacleState(obstacle_obs);

  // Observations
  obs << goal_linear_vel_, ori, quad_state_.v, obstacle_obs;
  return true;
}

bool VisionEnv::getObstacleState(Ref<Vector<>> obs_state) {
  if (dynamic_objects_.size() <= 0 || static_objects_.size() <= 0) {
    logger_.error("No dynamic or static obstacles.");
    return false;
  }
  // make sure to reset the collision penalty
  relative_pos_norm_.clear();
  obstacle_radius_.clear();

  //ort_i
  quad_ptr_->getState(&quad_state_);

  // compute relative distance to dynamic obstacles
  std::vector<Vector<3>, Eigen::aligned_allocator<Vector<3>>> relative_pos;
  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    // compute relative position vector
    Vector<3> delta_pos = dynamic_objects_[i]->getPos() - quad_state_.p;
    relative_pos.push_back(delta_pos);

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();
    // limit observation range
    if (obstacle_dist > max_detection_range_) {
      obstacle_dist = max_detection_range_;
    }
    relative_pos_norm_.push_back(obstacle_dist);

    // store the obstacle radius
    Scalar obs_radius = dynamic_objects_[i]->getScale()[0] / 2;
    obstacle_radius_.push_back(obs_radius);

    //
    if (obstacle_dist < obs_radius) {
      is_collision_ = true;
    }
  }

  // compute relatiev distance to static obstacles
  for (int i = 0; i < (int)static_objects_.size(); i++) {
    // compute relative position vector
    Vector<3> delta_pos = static_objects_[i]->getPos() - quad_state_.p;
    relative_pos.push_back(delta_pos);

    // compute relative distance
    Scalar obstacle_dist = delta_pos.norm();
    if (obstacle_dist > max_detection_range_) {
      obstacle_dist = max_detection_range_;
    }
    relative_pos_norm_.push_back(obstacle_dist);

    // store the obstacle radius
    Scalar obs_radius = static_objects_[i]->getScale()[0] / 2;
    obstacle_radius_.push_back(obs_radius);

    if (obstacle_dist < obs_radius) {
      is_collision_ = true;
    }
  }

  // std::cout << relative_pos_norm_ << std::endl;
  size_t idx = 0;
  for (size_t sort_idx : sort_indexes(relative_pos_norm_)) {
    if (idx >= visionenv::kNObstacles) break;

    if (idx < relative_pos.size()) {
      // if enough obstacles in the environment
      if (relative_pos_norm_[sort_idx] <= max_detection_range_) {
        // if obstacles are within detection range
        obs_state.segment<visionenv::kNObstaclesState>(
          idx * visionenv::kNObstaclesState)
          << relative_pos[sort_idx],
          obstacle_radius_[sort_idx];
      } else {
        // if obstacles are beyong detection range
        obs_state.segment<visionenv::kNObstaclesState>(
          idx * visionenv::kNObstaclesState) =
          Vector<4>(max_detection_range_, max_detection_range_,
                    max_detection_range_, obstacle_radius_[sort_idx]);
      }

    } else {
      // if not enough obstacles in the environment
      obs_state.segment<visionenv::kNObstaclesState>(
        idx * visionenv::kNObstaclesState) =
        Vector<visionenv::kNObstaclesState>(max_detection_range_,
                                            max_detection_range_,
                                            max_detection_range_, 0.0);
    }
    idx += 1;
  }

  return true;
}

bool VisionEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs,
                     Ref<Vector<>> reward) {
  if (!act.allFinite() || act.rows() != act_dim_ || rew_dim_ != reward.rows()) {
    return false;
    logger_.error(
      "Cannot run environment simulation. dimension mismatch or invalid "
      "actions.");
  }

  //
  old_pi_act_ = pi_act_;

  // compute actual control actions
  // act has range between [-1, 1] due to Tanh layer of the NN policy
  pi_act_ = act.cwiseProduct(act_std_) + act_mean_;

  cmd_.t += sim_dt_;
  quad_state_.t += sim_dt_;

  // apply old actions to simulate delay
  cmd_.collective_thrust = old_pi_act_(0);
  cmd_.omega = old_pi_act_.segment<3>(1);

  // simulate quadrotor
  quad_ptr_->run(cmd_, sim_dt_);

  // update quadrotor state and old quad_state
  quad_old_state_ = quad_state_;
  quad_ptr_->getState(&quad_state_);

  // simulate dynamic obstacles
  simDynamicObstacles(sim_dt_);

  // update observations
  getObs(obs);

  return computeReward(reward);
}

bool VisionEnv::simDynamicObstacles(const Scalar dt) {
  if (dynamic_objects_.size() <= 0) {
    logger_.warn(
      "No Dynamic Obstacles defined. Skipping dynamic obstacles simulation.");
    return false;
  }
  for (int i = 0; i < int(dynamic_objects_.size()); i++) {
    dynamic_objects_[i]->run(sim_dt_);
  }
  return true;
}
//
// std::vector<Scalar> Minus(std::vector<float> v1, std::vector<float> v2){
//     std::vector<Scalar> v;
//     for(int i = 0; i < v1.size(); i++){
//         v[i] = v2[i] - v1[i];
//     }
//     return v;
// }

static double getDistance(Vector<3> v1, Vector<3> v2){
  double len = 0;
  for(int i = 0; i < 3; i++){
      len += (v2[i] - v1[i]) * (v2[i] - v1[i]);
  }
  return sqrt(len);
}

static Scalar dotProduct(Vector<3> vect_A, Vector<3> vect_B)
{

    Scalar product = 0;
    // Loop for calculate dot product
    for (int i = 0; i < 3; i++)
        product = product + vect_A[i] * vect_B[i];
    return product;
}

bool VisionEnv::computeReward(Ref<Vector<>> reward) {
  // ---------------------- reward function design

  const Scalar velocityX =quad_state_.x(QS::VELX);
  const Scalar velocityY =quad_state_.x(QS::VELY);
  const Scalar velocityZ =quad_state_.x(QS::VELZ);

 const Scalar velX = velocityX > 0 ? abs(velocityX) : 0;

  logger_.error( to_string(velocityX) + " " + to_string(velocityY) + " " +  to_string(velocityZ) );
  //logger_.error( to_string(positionX) + " " + to_string(positionY) + " " +  to_string(positionZ) );
  //logger_.error( to_string(quad_state_.p.norm()) );
  //logger_.error( to_string(quad_state_.p[0]) + " " + to_string(quad_state_.p[1]) + " " +  to_string(quad_state_.p[2]) );

    Scalar dist_reward = 2 * sqrt(velX) * (1.0 - sqrt(abs(goal_pos_[0] -  quad_state_.p(QS::POSX)) / abs(max_dist_[0])));

   Scalar time_percentage = (max_t_ - cmd_.t) / max_t_;

  // - angular velocity penalty, to avoid oscillations

  // const Scalar positionX =quad_state_.x(QS::POSX);
  // const Scalar positionY =quad_state_.x(QS::POSY);
  // const Scalar positionZ =quad_state_.x(QS::POSZ);


  string gg = "";

 Eigen::Vector3d pos_new(quad_state_.p(QS::POSX), quad_state_.p(QS::POSY), quad_state_.p(QS::POSZ));
 Eigen::Vector3d pos_old(quad_old_state_.p(QS::POSX), quad_old_state_.p(QS::POSY), quad_old_state_.p(QS::POSZ));
Eigen::Vector3d velocity_vec = quad_state_.v;
 Eigen::Vector3d drone_dir;
 Eigen::Vector3d WorldX(1,0,0);
 if(drone_orientation_.compare("global") == 0){
      drone_dir = WorldX;
      gg = "global";
 }else {

        drone_dir = velocity_vec / quad_state_.v.norm();
      gg = "local";
 }

Eigen::Matrix3d rot_mat = quad_state_.R();
Eigen::Vector3d origin(1,0,0);
Eigen::Vector3d camera_dir =  rot_mat * origin;
//Scalar attitude_reward = attitude_ori_coeff_ * tanh(2.2 * drone_dir.dot(camera_dir));
Scalar attitude_reward = 0.6 * log(velX + 1) * tanh(1.1 * drone_dir.dot(camera_dir));

logger_.warn( to_string( drone_dir.dot(camera_dir) ));


 // get N most closest obstacles as the observation
 Vector<visionenv::kNObstacles * visionenv::kNObstaclesState> obstacles;
 getObstacleState(obstacles);

 // - compute collision penalty idea di giuseppe 2
  Scalar collision_penalty = 0.0;
  Scalar max_collision_penalty = 0.0;
  //Scalar total_detectable_obstacles = 0;
  //size_t idx = 0;
 //logger_.warn(  to_string(obstacles.size()) );
  for (int i = 0; i < obstacles.size() / 4; i++){
      //per single obstacle
      Eigen::Vector3d ob_relative_pos(obstacles[i*4+0], obstacles[i*4+1], obstacles[i*4+2]);
      Eigen::Vector3d ori(0,0,0);
      Scalar radius = obstacles[i*4+3];
      Scalar obstacle_dis = getDistance(ori, ob_relative_pos);
      obstacle_dis = (obstacle_dis > 0) && (obstacle_dis < max_detection_range_) ?
                    obstacle_dis : max_detection_range_;
      Eigen::Vector3d obs_dir = ob_relative_pos / obstacle_dis;
      Scalar dot_theta = abs(obs_dir.dot(drone_dir));
      Scalar theta = acos(dot_theta); // the magnitude of these 2 dirs are one.
      theta = theta * 180 / abs(obstacle_dis * 3.141592653589793);
      Scalar alpha = radius * 180 / abs(obstacle_dis * 3.141592653589793);

      if(theta < alpha && theta == theta && alpha == alpha){ //non modificare pls
          // collision_penalty = (1 / 10 - 1 / obstacle_dis);
          // if(collision0_penalty < -1 ){
          //   collision_penalty = -1;
          //collision_penalty -= 1/(0.1 * pow(obstacle_dis, 8) + 1);
          //collision_penalty -= 1/( 1/velX * pow(obstacle_dis, 5) + 1);
          Scalar soft_range = 0.5; // 2
        Scalar hard_range = 0;
        Eigen::Vector3d linear_acceleration = quad_state_.a;
        Scalar acc_module = linear_acceleration.dot(obs_dir);
        Scalar vel_module = velocity_vec.dot(obs_dir);
        hard_range = acc_module > 0 ? vel_module * vel_module / (2 * acc_module) : 0;

        collision_penalty = - 1 /(  abs(pow( (pow(1.2*hard_range, 2.6) + soft_range), -2.4)
                                      * pow(obstacle_dis, 6.5)) + 1);

        if(max_collision_penalty > collision_penalty){
            max_collision_penalty = collision_penalty;
        }
      }
      //collision_penalty *= collision_coeff_;

    //  if(theta != theta || alpha != alpha)
      //logger_.warn(  to_string(i) + " " +  to_string(theta) + " " + to_string(alpha) );
  }
  collision_penalty = max_collision_penalty;
  //idea giuseppe
  // if(quad_state_.p(QS::POSX) > xMax){
  //   dist_reward = dist_reward *(1 + codrone_dir.dot(camera_dir)llision_penalty);
  //   xMax =  quad_state_.p(QS::POSX);
  // }else{
  //   dist_reward = 0;
  // }

  //logger_.info( "angular velocit: " + str1 + " " + str2 + " " + str3);
 // logger_.info( "attitude : " + gg);

 Scalar Wall_behind_penalty = velocityX > survive_rew_ ? 0 : velocityX / 30;

  //  change progress reward as survive reward
   Scalar total_reward =
         dist_reward + collision_penalty + attitude_reward + Wall_behind_penalty;
  string str =to_string(dist_reward) + "  " + to_string(collision_penalty)
                  +  "  " +
    to_string(attitude_reward) +  "  " + to_string(Wall_behind_penalty)
    +  "  " + to_string(total_reward) ;
//  logger_.warn(str);
  //ogger_.info(to_string(( num_dynamic_objects_ + num_static_objects_ )));


  //
  // if(quad_state_.p(QS::POSX) > xMax)
  //    xMax =  quad_state_.p(QS::POSX);
  //
  // if (xMax - 0.5 > quad_state_.p(QS::POSX)){
  //   total_reward = -0.5;
  // }



    // return all reward components for debug purposes
    reward << dist_reward, collision_penalty, attitude_reward, survive_rew_, total_reward;
  return true;
}

bool VisionEnv::isTerminalState(Scalar &reward) {

 Scalar time_percentage = (max_t_ - cmd_.t) / max_t_;
 Scalar maxCollision = num_dynamic_objects_ + num_static_objects_;

  //collsion
  if (is_collision_) {
      reward = -1.0; //-1.0;
      std::cout << "Collision!\n";
      return true;
  }

 //simulation time out
 if (cmd_.t >= max_t_ - sim_dt_) {
   reward =  -10.0; //-1;
   std::cout << "Timeout!\n";
   return true;
 }

  // world boundling box check
  // - x, y, and z
  const Scalar safty_threshold = 0.1;
  bool x_valid = quad_state_.p(QS::POSX) >= world_box_[0] + safty_threshold &&
                 quad_state_.p(QS::POSX) <= world_box_[1] - safty_threshold;
  bool y_valid = quad_state_.p(QS::POSY) >= world_box_[2] + safty_threshold &&
                 quad_state_.p(QS::POSY) <= world_box_[3] - safty_threshold;
  bool z_valid = quad_state_.x(QS::POSZ) >= world_box_[4] + safty_threshold &&
                 quad_state_.x(QS::POSZ) <= world_box_[5] - safty_threshold;
 if (!x_valid || !y_valid || !z_valid) {
    reward = -1.0;//-1.0;
    return true;
  }


//for this competitio, only evaluate x position
if (abs(goal_pos_[0] - quad_state_.p(QS::POSX)) < 10){
  reward = 10.0;
  //Scalar collisionPercentage = (maxCollision - num_collision) / (maxCollision);
  //reward = reward;
  std::cout << "reached target position!\n";
  return true;
}

  return false;
}


bool VisionEnv::getQuadAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && pi_act_.allFinite() && (act.size() == pi_act_.size())) {
    act = pi_act_;
    return true;
  }
  return false;
}

bool VisionEnv::getQuadState(Ref<Vector<>> obs) const {
  if (quad_state_.t >= 0.0 && (obs.rows() == visionenv::kNQuadState)) {
    obs << quad_state_.t, quad_state_.p, quad_state_.qx, quad_state_.v,
      quad_state_.w, quad_state_.a, quad_ptr_->getMotorOmega(),
      quad_ptr_->getMotorThrusts();
    return true;
  }
  logger_.error("Get Quadrotor state failed.");
  return false;
}

bool VisionEnv::getDepthImage(Ref<DepthImgVector<>> depth_img) {
  if (!rgb_camera_ || !rgb_camera_->getEnabledLayers()[0]) {
    logger_.error(
      "No RGB Camera or depth map is not enabled. Cannot retrieve depth "
      "images.");
    return false;
  }
  rgb_camera_->getDepthMap(depth_img_);

  depth_img = Map<DepthImgVector<>>((float_t *)depth_img_.data,
                                    depth_img_.rows * depth_img_.cols);
  return true;
}


bool VisionEnv::getImage(Ref<ImgVector<>> img, const bool rgb) {
  if (!rgb_camera_) {
    logger_.error("No Camera! Cannot retrieve Images.");
    return false;
  }

  rgb_camera_->getRGBImage(rgb_img_);

  if (rgb_img_.rows != img_height_ || rgb_img_.cols != img_width_) {
    logger_.error(
      "Image resolution mismatch. Aborting.. Image rows %d != %d, Image cols "
      "%d != %d",
      rgb_img_.rows, img_height_, rgb_img_.cols, img_width_);
    return false;
  }

  if (!rgb) {
    // converting rgb image to gray image
    cvtColor(rgb_img_, gray_img_, CV_RGB2GRAY);
    // map cv::Mat data to Eiegn::Vector
    img = Map<ImgVector<>>(gray_img_.data, gray_img_.rows * gray_img_.cols);
  } else {
    img = Map<ImgVector<>>(rgb_img_.data, rgb_img_.rows * rgb_img_.cols *
                                            rgb_camera_->getChannels());
  }
  return true;
}


bool VisionEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["environment"]) {
    difficulty_level_ = cfg["environment"]["level"].as<std::string>();
    env_folder_ = cfg["environment"]["env_folder"].as<std::string>();
    world_box_ = cfg["environment"]["world_box"].as<std::vector<Scalar>>();
    std::vector<Scalar> goal_vel_vec =
      cfg["environment"]["goal_vel"].as<std::vector<Scalar>>();
    goal_linear_vel_ = Vector<3>(goal_vel_vec.data());
    //my code:
    std::vector<Scalar> goal_pos_vec =
      cfg["environment"]["goal_pos"].as<std::vector<Scalar>>();
       goal_pos_ = Vector<3>(goal_pos_vec.data());

    std::vector<Scalar> ref_qx =
      cfg["environment"]["ref_qx"].as<std::vector<Scalar>>();
      ref_qx_= Vector<4>(ref_qx.data());

    max_detection_range_ =
      cfg["environment"]["max_detection_range"].as<Scalar>();
  }

  if (cfg["simulation"]) {
    sim_dt_ = cfg["simulation"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["simulation"]["max_t"].as<Scalar>();
    quadrotorScale_ = cfg["simulation"]["fakeQuadrotorScaleSize"].as<float>();

  } else {
    logger_.error("Cannot load [quadrotor_env] parameters");
    return false;
  }

  if (cfg["rewards"]) {
    // load reward coefficients for reinforcement learning
    vel_coeff_ = cfg["rewards"]["vel_coeff"].as<Scalar>();
    collision_coeff_ = cfg["rewards"]["collision_coeff"].as<Scalar>();
    angular_vel_coeff_ = cfg["rewards"]["angular_vel_coeff"].as<Scalar>();
    survive_rew_ = cfg["rewards"]["survive_rew"].as<Scalar>();
    goal_dist_rew_ = cfg["rewards"]["goal_dist_rew"].as<Scalar>();
    attitude_ori_coeff_ = cfg["rewards"]["attitude_ori_coeff"].as<Scalar>();
    drone_orientation_ = cfg["rewards"]["orientation"].as<std::string>();

    // load reward settings
    reward_names_ = cfg["rewards"]["names"].as<std::vector<std::string>>();
    rew_dim_ = reward_names_.size();
  } else {
    logger_.error("Cannot load [rewards] parameters");
    return false;
  }

  // environment
  if (cfg["unity"]) {
    unity_render_ = cfg["unity"]["render"].as<bool>();
    scene_id_ = cfg["unity"]["scene_id"].as<SceneID>();
  }

  //
  std::string scene_file =
    getenv("FLIGHTMARE_PATH") + std::string("/flightpy/configs/scene.yaml");
  // check if configuration file exist
  if (!(file_exists(scene_file))) {
    logger_.error("Unity scene configuration file %s does not exists.",
                  scene_file);
  }
  // load configuration file
  YAML::Node scene_cfg_node = YAML::LoadFile(scene_file);
  std::string scene_idx = "scene_" + std::to_string(scene_id_);

  std::vector<Scalar> render_offset =
    scene_cfg_node[scene_idx]["render_offset"].as<std::vector<Scalar>>();
  unity_render_offset_ = Vector<3>(render_offset.data());
  return true;
}

bool VisionEnv::configDynamicObjects(const std::string &yaml_file) {
  //
  if (!(file_exists(yaml_file))) {
    logger_.error("Configuration file %s does not exists.", yaml_file);
    return false;
  }
  YAML::Node cfg_node = YAML::LoadFile(yaml_file);

  // logger_.info("Configuring dynamic objects");

  int num_objects = cfg_node["N"].as<int>();
  // create static objects
  for (int i = 0; i < num_objects; i++) {

    std::string object_id = "Object" + std::to_string(i + 1);
    std::string prefab_id = cfg_node[object_id]["prefab"].as<std::string>();
  //  prefab_id = "rpg_box0" + to_string(i % 3 + 1);
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    // load location, rotation and size
    std::vector<Scalar> posvec =
      (cfg_node[object_id]["position"]).as<std::vector<Scalar>>();
    std::vector<Scalar> rotvec =
      (cfg_node[object_id]["rotation"]).as<std::vector<Scalar>>();
    std::vector<Scalar> scalevec =
      (cfg_node[object_id]["scale"]).as<std::vector<Scalar>>();

    obj->setPosition(Vector<3>(posvec.data()));
    obj->setRotation(Quaternion(rotvec.data()));
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(Vector<3>(scalevec.data()));

    std::string csv_name = cfg_node[object_id]["csvtraj"].as<std::string>();
    std::string csv_file = obstacle_cfg_path_ + std::string("/csvtrajs/") +
                           csv_name + std::string(".csv");
    if (!(file_exists(csv_file))) {
      logger_.error("Configuration file %s does not exists.", csv_file);
      return false;
    }
    obj->loadTrajectory(csv_file);

    dynamic_objects_.push_back(obj);
  }
  num_dynamic_objects_ = dynamic_objects_.size();
  return true;
}

bool VisionEnv::configStaticObjects(const std::string &csv_file) {
  //
  if (!(file_exists(csv_file))) {
    logger_.error("Configuration file %s does not exists.", csv_file);
    return false;
  }
  std::ifstream infile(csv_file);
  int i = 0;
  for (auto &row : CSVRange(infile)) {
    // Read column 0 for time
    std::string object_id = "StaticObject" + std::to_string(i + 1);
    std::string prefab_id = (std::string)row[0];
    //prefab_id = "rpg_box03";
    //
    std::shared_ptr<UnityObject> obj =
      std::make_shared<UnityObject>(object_id, prefab_id);

    //
    Vector<3> pos;
    pos << std::stod((std::string)row[1]), std::stod((std::string)row[2]),
      std::stod((std::string)row[3]);

    Quaternion quat;
    quat.w() = std::stod((std::string)row[4]);
    quat.x() = std::stod((std::string)row[5]);
    quat.y() = std::stod((std::string)row[6]);
    quat.z() = std::stod((std::string)row[7]);

    Vector<3> scale;
    scale << std::stod((std::string)row[8]), std::stod((std::string)row[9]),
      std::stod((std::string)row[10]);

    //
    obj->setPosition(pos);
    obj->setRotation(quat);
    // actual size in meters
    obj->setSize(Vector<3>(1.0, 1.0, 1.0));
    // scale of the original size
    obj->setScale(scale);
    static_objects_.push_back(obj);
  }
  num_static_objects_ = static_objects_.size();

  return true;
}

bool VisionEnv::configCamera(const YAML::Node &cfg) {
  if (!cfg["rgb_camera"]) {
    logger_.error("Cannot config RGB Camera");
    return false;
  }

  if (!cfg["rgb_camera"]["on"].as<bool>()) {
    logger_.warn("Camera is off. Please turn it on.");
    return false;
  }

  if (quad_ptr_->getNumCamera() >= 1) {
    logger_.warn("Camera has been added. Skipping the camera configuration.");
    return false;
  }

  // create camera
  rgb_camera_ = std::make_shared<RGBCamera>();

  // load camera settings
  std::vector<Scalar> t_BC_vec =
    cfg["rgb_camera"]["t_BC"].as<std::vector<Scalar>>();
  std::vector<Scalar> r_BC_vec =
    cfg["rgb_camera"]["r_BC"].as<std::vector<Scalar>>();

  //
  Vector<3> t_BC(t_BC_vec.data());
  Matrix<3, 3> r_BC =
    (AngleAxis(r_BC_vec[2] * M_PI / 180.0, Vector<3>::UnitZ()) *
     AngleAxis(r_BC_vec[1] * M_PI / 180.0, Vector<3>::UnitY()) *
     AngleAxis(r_BC_vec[0] * M_PI / 180.0, Vector<3>::UnitX()))
      .toRotationMatrix();
  std::vector<bool> post_processing = {false, false, false};
  post_processing[0] = cfg["rgb_camera"]["enable_depth"].as<bool>();
  post_processing[1] = cfg["rgb_camera"]["enable_segmentation"].as<bool>();
  post_processing[2] = cfg["rgb_camera"]["enable_opticalflow"].as<bool>();

  //
  rgb_camera_->setFOV(cfg["rgb_camera"]["fov"].as<Scalar>());
  rgb_camera_->setWidth(cfg["rgb_camera"]["width"].as<int>());
  rgb_camera_->setChannels(cfg["rgb_camera"]["channels"].as<int>());
  rgb_camera_->setHeight(cfg["rgb_camera"]["height"].as<int>());
  rgb_camera_->setRelPose(t_BC, r_BC);
  rgb_camera_->setPostProcessing(post_processing);


  // add camera to the quadrotor
  quad_ptr_->addRGBCamera(rgb_camera_);

  // adapt parameters
  img_width_ = rgb_camera_->getWidth();
  img_height_ = rgb_camera_->getHeight();
  rgb_img_ = cv::Mat::zeros(img_height_, img_width_,
                            CV_MAKETYPE(CV_8U, rgb_camera_->getChannels()));
  depth_img_ = cv::Mat::zeros(img_height_, img_width_, CV_32FC1);
  return true;
}

bool VisionEnv::addQuadrotorToUnity(const std::shared_ptr<UnityBridge> bridge) {
  if (!quad_ptr_) return false;

  bridge->addQuadrotor(quad_ptr_, quadrotorScale_);
  for (int i = 0; i < (int)dynamic_objects_.size(); i++) {
    bridge->addDynamicObject(dynamic_objects_[i]);
  }

  //
  bridge->setRenderOffset(unity_render_offset_);
  bridge->setObjectCSV(static_object_csv_);
  return true;
}

void VisionEnv::setFakeQuadrotorScale(float scale){
  quadrotorScale_ = scale;
}

bool VisionEnv::setUnity(bool render, const int input_port, const int output_port) {
  unity_render_ = render;
  if (!unity_render_ || unity_bridge_ptr_ != nullptr) {
    logger_.warn(
      "Unity render is False or Flightmare Bridge has been already created. "
      "Cannot set Unity.");
    return false;
  }
  // create unity bridge
  unity_bridge_ptr_ = UnityBridge::getInstance(input_port, output_port);
  // add objects to Unity

  addQuadrotorToUnity(unity_bridge_ptr_);

  logger_.info("Flightmare Bridge created.");
  return true;
}

bool VisionEnv::setUnity(bool render) {
  unity_render_ = render;
  if (!unity_render_ || unity_bridge_ptr_ != nullptr) {
    logger_.warn(
      "Unity render is False or Flightmare Bridge has been already created. "
      "Cannot set Unity.");
    return false;
  }
  // create unity bridge
  unity_bridge_ptr_ = UnityBridge::getInstance();
  // add objects to Unity

  addQuadrotorToUnity(unity_bridge_ptr_);

  logger_.info("Flightmare Bridge created. Yeah!");
  return true;
}


bool VisionEnv::connectUnity(void) {
  if (unity_bridge_ptr_ == nullptr) return false;
  unity_ready_ = unity_bridge_ptr_->connectUnity(scene_id_);
  return unity_ready_;
}


FrameID VisionEnv::updateUnity(const FrameID frame_id) {
  if (unity_render_ && unity_ready_) {
    unity_bridge_ptr_->getRender(frame_id);
    return unity_bridge_ptr_->handleOutput(frame_id);
  } else {
    return 0;
  }
}


void VisionEnv::disconnectUnity(void) {
  if (unity_bridge_ptr_ != nullptr) {
    unity_bridge_ptr_->disconnectUnity();
    unity_ready_ = false;
  } else {
    logger_.warn("Flightmare Unity Bridge is not initialized.");
  }
}

int VisionEnv::getNumDetectedObstacles(void) { return num_detected_obstacles_; }

std::ostream &operator<<(std::ostream &os, const VisionEnv &vision_env) {
  os.precision(3);
  os << "Vision Environment:\n"
     << "obs dim =            [" << vision_env.obs_dim_ << "]\n"
     << "act dim =            [" << vision_env.act_dim_ << "]\n"
     << "#dynamic objects=    [" << vision_env.num_dynamic_objects_ << "]\n"
     << "#static objects=     [" << vision_env.num_static_objects_ << "]\n"
     << "obstacle dim =       [" << vision_env.num_detected_obstacles_ << "]\n"
     << "sim dt =             [" << vision_env.sim_dt_ << "]\n"
     << "max_t =              [" << vision_env.max_t_ << "]\n"
     << "act_mean =           [" << vision_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << vision_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << vision_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << vision_env.obs_std_.transpose() << "]"
     << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
