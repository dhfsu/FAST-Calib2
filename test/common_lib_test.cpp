#include <gtest/gtest.h>

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "common_lib.h"

namespace
{
pcl::PointXYZ makePoint(const Eigen::Vector3f& p)
{
  pcl::PointXYZ out;
  out.x = p.x();
  out.y = p.y();
  out.z = p.z();
  return out;
}

Eigen::Vector3f toOptical(const Eigen::Vector3f& native,
                         const Eigen::Vector3f& forward,
                         const Eigen::Vector3f& left,
                         const Eigen::Vector3f& up)
{
  return Eigen::Vector3f(-native.dot(left),
                         -native.dot(up),
                          native.dot(forward));
}
}  // namespace

TEST(LidarMountAxes, ResolvesAll24RightHandedCombinations)
{
  const std::vector<std::string> axes = {"+x", "-x", "+y", "-y", "+z", "-z"};
  const std::array<Eigen::Vector3f, 4> body_points = {{
      Eigen::Vector3f(3.0f,  0.25f,  0.20f),
      Eigen::Vector3f(3.0f, -0.25f, -0.20f),
      Eigen::Vector3f(3.0f,  0.25f, -0.20f),
      Eigen::Vector3f(3.0f, -0.25f,  0.20f)}};

  int valid_combinations = 0;
  for (const auto& forward_name : axes)
  {
    for (const auto& up_name : axes)
    {
      Eigen::Vector3f forward;
      Eigen::Vector3f left;
      Eigen::Vector3f up;
      std::string normalized_forward;
      std::string normalized_left;
      std::string normalized_up;
      std::string error;
      if (!resolveLidarMountAxes(forward_name, up_name,
                                 forward, left, up,
                                 normalized_forward, normalized_left, normalized_up,
                                 error))
      {
        continue;
      }

      ++valid_combinations;
      EXPECT_NEAR(forward.dot(up), 0.0f, 1e-6f);
      EXPECT_NEAR((forward.cross(left) - up).norm(), 0.0f, 1e-6f);

      pcl::PointCloud<pcl::PointXYZ>::Ptr camera_points(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_points(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& body : body_points)
      {
        const Eigen::Vector3f native =
            forward * body.x() + left * body.y() + up * body.z();
        lidar_points->push_back(makePoint(native));
        camera_points->push_back(makePoint(Eigen::Vector3f(-body.y(), -body.z(), body.x())));
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_camera(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_lidar(new pcl::PointCloud<pcl::PointXYZ>);
      ASSERT_TRUE(sortPatternCenters(camera_points, sorted_camera, "camera"));
      ASSERT_TRUE(sortPatternCenters(lidar_points, sorted_lidar, "lidar",
                                     forward_name, up_name));

      ASSERT_EQ(sorted_camera->size(), sorted_lidar->size());
      for (std::size_t i = 0; i < sorted_lidar->size(); ++i)
      {
        const auto& p = sorted_lidar->points[i];
        const Eigen::Vector3f optical =
            toOptical(Eigen::Vector3f(p.x, p.y, p.z), forward, left, up);
        EXPECT_NEAR(optical.x(), sorted_camera->points[i].x, 1e-5f);
        EXPECT_NEAR(optical.y(), sorted_camera->points[i].y, 1e-5f);
        EXPECT_NEAR(optical.z(), sorted_camera->points[i].z, 1e-5f);
      }
    }
  }

  EXPECT_EQ(valid_combinations, 24);
}

TEST(LidarMountAxes, RejectsInvalidOrParallelAxes)
{
  std::string error;
  EXPECT_FALSE(validateLidarMountAxes("+x", "-x", error));
  EXPECT_FALSE(validateLidarMountAxes("front", "+z", error));
  EXPECT_FALSE(validateLidarMountAxes("+x", "", error));
}

TEST(CameraCalibration, RequiresMatchingResolution)
{
  Params params{};
  params.camera_width = 2448;
  params.camera_height = 2048;
  params.fx = 2364.0;
  params.fy = 2368.0;
  params.cx = 1211.0;
  params.cy = 1040.0;
  params.k1 = -0.05;
  params.k2 = 0.12;
  params.p1 = 0.0;
  params.p2 = 0.0;

  std::string error;
  EXPECT_TRUE(validateCameraCalibrationForImage(params, 2448, 2048, error));
  EXPECT_FALSE(validateCameraCalibrationForImage(params, 1224, 1024, error));

  params.camera_width = 0;
  EXPECT_FALSE(validateCameraCalibrationForImage(params, 2448, 2048, error));
}

TEST(OutputDirectory, CreatesMissingParents)
{
  const std::string root =
      "/tmp/fast_calib_common_lib_test_" + std::to_string(static_cast<long long>(::getpid()));
  const std::string first = root + "/first";
  const std::string nested = first + "/second";

  std::string error;
  ASSERT_TRUE(ensureDirectoryTree(nested, error)) << error;

  struct stat status;
  ASSERT_EQ(::stat(nested.c_str(), &status), 0);
  EXPECT_TRUE(S_ISDIR(status.st_mode));

  EXPECT_EQ(::rmdir(nested.c_str()), 0);
  EXPECT_EQ(::rmdir(first.c_str()), 0);
  EXPECT_EQ(::rmdir(root.c_str()), 0);
}


namespace
{
// 用户实际使用的鱼眼相机（vanjee，4000x3000，Kannala-Brandt k1..k4）
constexpr int kFisheyeWidth = 4000;
constexpr int kFisheyeHeight = 3000;
constexpr double kFisheyeFx = 792.4057085207388;
constexpr double kFisheyeFy = 792.3582580692001;
constexpr double kFisheyeCx = 2004.5367643055422;
constexpr double kFisheyeCy = 1500.454819537657;
constexpr double kFisheyeK[4] = {0.09095816477283435, -0.02448764621169176,
                                 0.008319662193425007, -0.0036367895734679048};

Params makeFisheyeParams()
{
  Params params{};
  params.camera_model = "fisheye";
  params.camera_width = kFisheyeWidth;
  params.camera_height = kFisheyeHeight;
  params.fx = kFisheyeFx;
  params.fy = kFisheyeFy;
  params.cx = kFisheyeCx;
  params.cy = kFisheyeCy;
  params.k1 = kFisheyeK[0];
  params.k2 = kFisheyeK[1];
  params.k3 = kFisheyeK[2];
  params.k4 = kFisheyeK[3];
  params.fisheye_max_theta_deg = 89.0;
  params.marker_size = 0.2;
  params.delta_width_qr_center = 0.55;
  params.delta_height_qr_center = 0.35;
  params.delta_width_circles = 0.5;
  params.delta_height_circles = 0.4;
  return params;
}

cv::Mat fisheyeCameraMatrix(const Params& params)
{
  return (cv::Mat_<double>(3, 3) << params.fx, 0, params.cx,
                                    0, params.fy, params.cy,
                                    0, 0, 1);
}

cv::Mat fisheyeDistCoeffs(const Params& params)
{
  return (cv::Mat_<double>(4, 1) << params.k1, params.k2, params.k3, params.k4);
}

int removeFileIfExists(const std::string& path)
{
  return ::remove(path.c_str());
}

// 读出 "key: value" 形式的数值（输出文件用 ostream 默认精度，故按值比较）
double extractYamlValue(const std::string& text, const std::string& key)
{
  const std::string needle = "\n" + key + ": ";
  const size_t pos = text.find(needle);
  if (pos == std::string::npos) return std::numeric_limits<double>::quiet_NaN();
  return std::stod(text.substr(pos + needle.size()));
}
}  // namespace

TEST(CameraModel, NormalizesUserSpellings)
{
  EXPECT_EQ(parseCameraModel(""), CameraModel::Pinhole);
  EXPECT_EQ(parseCameraModel("  "), CameraModel::Pinhole);
  EXPECT_EQ(parseCameraModel("pinhole"), CameraModel::Pinhole);
  EXPECT_EQ(parseCameraModel("Pinhole"), CameraModel::Pinhole);

  EXPECT_EQ(parseCameraModel("fisheye"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("FishEye"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel(" fish-eye "), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("equidistant"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("EquidistantCamera"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("equidistant_camera"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("kb4"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("KannalaBrandt"), CameraModel::Fisheye);
  EXPECT_EQ(parseCameraModel("OPENCV_FISHEYE"), CameraModel::Fisheye);

  // 其它模型必须被拒绝，而不是被当成针孔静默处理
  EXPECT_EQ(parseCameraModel("omni"), CameraModel::Unknown);
  EXPECT_EQ(parseCameraModel("ocam"), CameraModel::Unknown);
  EXPECT_EQ(parseCameraModel("atan"), CameraModel::Unknown);
  EXPECT_EQ(parseCameraModel("polynomial"), CameraModel::Unknown);
}

TEST(CameraCalibration, ValidatesFisheyeConfiguration)
{
  Params params = makeFisheyeParams();
  std::string error;

  EXPECT_TRUE(isFisheyeCameraModel(params));
  EXPECT_TRUE(validateCameraCalibrationForImage(params, kFisheyeWidth,
                                                kFisheyeHeight, error)) << error;
  // 分辨率与标定分辨率不一致时必须拒绝
  EXPECT_FALSE(validateCameraCalibrationForImage(params, 2448, 2048, error));
  EXPECT_FALSE(validateCameraCalibrationForImage(params, 1224, 1024, error));

  // k1..k4 必须有限
  params.k4 = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(validateCameraCalibrationForImage(params, kFisheyeWidth,
                                                 kFisheyeHeight, error));
  EXPECT_NE(error.find("k1..k4"), std::string::npos);

  // 半径模型必须可逆
  params = makeFisheyeParams();
  params.k1 = -0.5;
  EXPECT_FALSE(fisheyeRadiusModelIsMonotonic(params.k1, params.k2, params.k3,
                                             params.k4, params.fisheye_max_theta_deg));
  EXPECT_FALSE(validateCameraCalibrationForImage(params, kFisheyeWidth,
                                                 kFisheyeHeight, error));

  // 未知模型直接报错
  params = makeFisheyeParams();
  params.camera_model = "omni";
  EXPECT_FALSE(validateCameraCalibrationForImage(params, kFisheyeWidth,
                                                 kFisheyeHeight, error));

  // 真实系数在 89 度内单调
  const Params real_params = makeFisheyeParams();
  EXPECT_TRUE(fisheyeRadiusModelIsMonotonic(real_params.k1, real_params.k2,
                                            real_params.k3, real_params.k4,
                                            real_params.fisheye_max_theta_deg));
  // 正向模型在 89 度处的半径，应明显小于该镜头画幅半对角线（四角是黑边）
  const double radius_limit = fisheyeRadiusLimit(real_params.k1, real_params.k2,
                                                 real_params.k3, real_params.k4, 89.0);
  const double corner_radius = std::hypot(kFisheyeCx / kFisheyeFx,
                                          kFisheyeCy / kFisheyeFy);
  EXPECT_LT(radius_limit, corner_radius);
  EXPECT_GT(radius_limit, 1.5);  // 有效视场仍远大于 90 度
}

TEST(FisheyeUndistort, DropsMarkersOutsideTheModelDomain)
{
  Params params = makeFisheyeParams();
  const cv::Mat K = fisheyeCameraMatrix(params);
  const cv::Mat D = fisheyeDistCoeffs(params);

  // marker 0：真实图像里的位置（离主点约 400 px）
  std::vector<cv::Point2f> inside = {cv::Point2f(1700.f, 1200.f),
                                     cv::Point2f(1780.f, 1200.f),
                                     cv::Point2f(1780.f, 1280.f),
                                     cv::Point2f(1700.f, 1280.f)};
  // marker 1：画幅四角（黑边，超出模型有效域）
  std::vector<cv::Point2f> outside = {cv::Point2f(20.f, 20.f),
                                      cv::Point2f(120.f, 20.f),
                                      cv::Point2f(120.f, 120.f),
                                      cv::Point2f(20.f, 120.f)};

  std::vector<std::vector<cv::Point2f>> corners = {inside, outside};
  std::vector<std::vector<cv::Point2f>> undistorted;
  std::vector<char> valid;

  ASSERT_TRUE(undistortFisheyeObservations(corners, K, D, params.fisheye_max_theta_deg,
                                           undistorted, valid));
  ASSERT_EQ(valid.size(), 2u);
  EXPECT_EQ(valid[0], 1);
  EXPECT_EQ(valid[1], 0);  // 整块丢弃，而不是丢个别角点
  ASSERT_EQ(undistorted[0].size(), 4u);
  EXPECT_TRUE(undistorted[1].empty());

  // 主点附近（归一化半径约 0）必须保留，且去畸变后仍落在 (cx, cy)
  const std::vector<cv::Point2f> center_px = {cv::Point2f(
      static_cast<float>(params.cx), static_cast<float>(params.cy))};
  std::vector<cv::Point2f> center_marker;
  for (int i = 0; i < 4; ++i) center_marker.push_back(center_px[0]);
  std::vector<std::vector<cv::Point2f>> center_undistorted;
  std::vector<char> center_valid;
  ASSERT_TRUE(undistortFisheyeObservations({center_marker}, K, D,
                                           params.fisheye_max_theta_deg,
                                           center_undistorted, center_valid));
  // 输入是 float 像素，比较时以 float 取整后的值为准
  EXPECT_NEAR(center_undistorted[0][0].x, center_px[0].x, 1e-3);
  EXPECT_NEAR(center_undistorted[0][0].y, center_px[0].y, 1e-3);
}

TEST(FisheyePose, RecoversKnownBoardPose)
{
  Params params = makeFisheyeParams();
  const cv::Mat K = fisheyeCameraMatrix(params);
  const cv::Mat D = fisheyeDistCoeffs(params);

  std::vector<std::vector<cv::Point3f>> boardCorners;
  std::vector<cv::Point3f> boardCircleCenters;
  std::vector<int> boardIds;
  buildTargetBoardGeometry(params.marker_size, params.delta_width_qr_center,
                           params.delta_height_qr_center, params.delta_width_circles,
                           params.delta_height_circles, boardCorners,
                           boardCircleCenters, boardIds);
  ASSERT_EQ(boardCorners.size(), 4u);
  ASSERT_EQ(boardCircleCenters.size(), 4u);
  ASSERT_EQ(boardIds.size(), 4u);

  cv::Ptr<cv::aruco::Dictionary> dictionary =
      cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  cv::Ptr<cv::aruco::Board> board =
      cv::aruco::Board::create(boardCorners, dictionary, boardIds);

  // 正对相机 + 大角度斜视两种姿态；斜视时平面 PnP 的镜像解最容易出现
  const std::vector<cv::Vec3d> gt_rvecs = {cv::Vec3d(0.05, -0.15, 0.12),
                                           cv::Vec3d(0.05, -0.96, 0.12)};
  for (const auto& gt_rvec : gt_rvecs)
  {
    const cv::Vec3d gt_tvec(0.10, -0.05, 2.50);

    // 用真实鱼眼模型把标定板投到原始鱼眼图（模拟 detectMarkers 的输出）
    std::vector<std::vector<cv::Point2f>> corners_raw;
    for (const auto& marker_corners : boardCorners)
    {
      std::vector<cv::Point2f> projected;
      cv::fisheye::projectPoints(marker_corners, projected, gt_rvec, gt_tvec, K, D);
      corners_raw.push_back(projected);
    }

    std::vector<std::vector<cv::Point2f>> corners_undistorted;
    std::vector<char> valid;
    ASSERT_TRUE(undistortFisheyeObservations(corners_raw, K, D,
                                             params.fisheye_max_theta_deg,
                                             corners_undistorted, valid));
    for (char v : valid) ASSERT_EQ(v, 1);

    std::vector<cv::Vec3d> marker_rvecs, marker_tvecs;
    cv::Vec3d rvec, tvec;
    ASSERT_TRUE(estimateBoardPoseFromMarkers(board, corners_undistorted, boardIds,
                                             params.marker_size, K,
                                             cv::Mat::zeros(4, 1, CV_64F),
                                             marker_rvecs, marker_tvecs, rvec, tvec))
        << "rvec_gt = " << gt_rvec;

    EXPECT_GT(tvec[2], 0.0);
    EXPECT_NEAR(tvec[0], gt_tvec[0], 1e-3);
    EXPECT_NEAR(tvec[1], gt_tvec[1], 1e-3);
    EXPECT_NEAR(tvec[2], gt_tvec[2], 1e-3);

    cv::Mat R_gt, R_est;
    cv::Rodrigues(gt_rvec, R_gt);
    cv::Rodrigues(rvec, R_est);
    const double angle_error_deg =
        cv::norm(R_gt.t() * R_est - cv::Mat::eye(3, 3, CV_64F)) * 180.0 / CV_PI;
    EXPECT_LT(angle_error_deg, 0.05);

    // 圆心几何：斜视下解必须在正确的分支上（镜像解同样是刚体变换，
    // 只能靠与真值比较来发现）
    cv::Mat board_transform = cv::Mat::eye(3, 4, CV_64F);
    R_est.copyTo(board_transform(cv::Rect(0, 0, 3, 3)));
    cv::Mat t = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
    t.copyTo(board_transform(cv::Rect(3, 0, 1, 3)));

    std::vector<cv::Point3d> centers;
    for (const auto& center : boardCircleCenters)
    {
      const cv::Mat X = (cv::Mat_<double>(4, 1) << center.x, center.y, center.z, 1.0);
      const cv::Mat Y = board_transform * X;
      centers.emplace_back(Y.at<double>(0), Y.at<double>(1), Y.at<double>(2));
    }
    EXPECT_NEAR(cv::norm(centers[0] - centers[1]), params.delta_width_circles, 1e-3);
    EXPECT_NEAR(cv::norm(centers[0] - centers[3]), params.delta_height_circles, 1e-3);
  }
}

TEST(FisheyeProjection, SkipsPointsBehindTheCameraAndKeepsCameraFrameCoordinates)
{
  // 小尺寸虚拟相机：p1 落在图像中心，p2 在相机后方，p3 投到图像外
  const cv::Mat K = (cv::Mat_<double>(3, 3) << 100.0, 0.0, 50.0,
                                               0.0, 100.0, 50.0,
                                               0.0, 0.0, 1.0);
  const cv::Mat D = cv::Mat::zeros(4, 1, CV_64F);

  cv::Mat image(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
  image.at<cv::Vec3b>(50, 50) = cv::Vec3b(10, 20, 30);  // BGR

  pcl::PointCloud<Common::Point>::Ptr cloud(new pcl::PointCloud<Common::Point>);
  const std::vector<Eigen::Vector3f> points = {Eigen::Vector3f(0.f, 0.f, 2.f),
                                               Eigen::Vector3f(0.f, 0.f, -1.f),
                                               Eigen::Vector3f(100.f, 0.f, 1.f)};
  for (const auto& p : points)
  {
    Common::Point pt;
    pt.x = p.x(); pt.y = p.y(); pt.z = p.z(); pt.intensity = 0.f;
    pt.ring = 0; pt.scan_id = 0;
    cloud->push_back(pt);
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  projectPointCloudToImageFisheye(cloud, Eigen::Matrix4f::Identity(), K, D, image,
                                  colored);

  // 只保留相机前方且落在图像内的点
  ASSERT_EQ(colored->size(), 1u);
  EXPECT_FLOAT_EQ(colored->points[0].x, 0.f);   // 坐标是相机系，不是像素
  EXPECT_FLOAT_EQ(colored->points[0].y, 0.f);
  EXPECT_FLOAT_EQ(colored->points[0].z, 2.f);
  EXPECT_EQ(colored->points[0].r, 30);          // BGR -> RGB
  EXPECT_EQ(colored->points[0].g, 20);
  EXPECT_EQ(colored->points[0].b, 10);
}

TEST(CalibrationOutput, WritesTheModelSpecificHeader)
{
  const std::string root =
      "/tmp/fast_calib_output_test_" + std::to_string(static_cast<long long>(::getpid()));
  const cv::Mat image(64, 128, CV_8UC3, cv::Scalar(0, 0, 0));

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB p;
  p.x = 0.f; p.y = 0.f; p.z = 2.f; p.r = 1; p.g = 2; p.b = 3;
  colored->push_back(p);

  auto readHeader = [&](const Params& params) {
    std::string error;
    EXPECT_TRUE(ensureDirectoryTree(params.output_path, error)) << error;
    saveCalibrationResults(params, Eigen::Matrix4f::Identity(), colored, image);
    std::ifstream in(params.output_path + "/single_calib_result.txt");
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
  };

  // 针孔输出保持原格式
  Params pinhole{};
  pinhole.output_path = root + "/pinhole";
  pinhole.camera_model = "pinhole";
  pinhole.fx = 2364.0; pinhole.fy = 2368.0; pinhole.cx = 1211.0; pinhole.cy = 1040.0;
  pinhole.k1 = -0.05; pinhole.k2 = 0.12; pinhole.p1 = 1e-4; pinhole.p2 = 6e-4;
  const std::string pinhole_text = readHeader(pinhole);
  EXPECT_NE(pinhole_text.find("cam_model: Pinhole\n"), std::string::npos);
  EXPECT_NE(pinhole_text.find("cam_d0: -0.05\n"), std::string::npos);
  EXPECT_NE(pinhole_text.find("cam_d3: 0.0006\n"), std::string::npos);
  EXPECT_EQ(pinhole_text.find("\nk1: "), std::string::npos);

  // 鱼眼输出写成 FAST-LIVO2 的 EquidistantCamera（读 k1..k4），不能再写 cam_d*
  Params fisheye = makeFisheyeParams();
  fisheye.output_path = root + "/fisheye";
  const std::string fisheye_text = readHeader(fisheye);
  EXPECT_NE(fisheye_text.find("cam_model: EquidistantCamera\n"), std::string::npos);
  EXPECT_NE(fisheye_text.find("cam_width: 128\n"), std::string::npos);
  EXPECT_NE(fisheye_text.find("cam_height: 64\n"), std::string::npos);
  EXPECT_TRUE(std::isnan(extractYamlValue(fisheye_text, "cam_d0")));
  EXPECT_NEAR(extractYamlValue(fisheye_text, "k3"), fisheye.k3, 1e-8);
  EXPECT_EQ(fisheye_text.find("cam_d0"), std::string::npos);

  // 清理
  removeFileIfExists(root + "/pinhole/single_calib_result.txt");
  removeFileIfExists(root + "/pinhole/colored_cloud.pcd");
  removeFileIfExists(root + "/pinhole/qr_detect.png");
  removeFileIfExists(root + "/fisheye/single_calib_result.txt");
  removeFileIfExists(root + "/fisheye/colored_cloud.pcd");
  removeFileIfExists(root + "/fisheye/qr_detect.png");
  EXPECT_EQ(::rmdir((root + "/pinhole").c_str()), 0);
  EXPECT_EQ(::rmdir((root + "/fisheye").c_str()), 0);
  EXPECT_EQ(::rmdir(root.c_str()), 0);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
