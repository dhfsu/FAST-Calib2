/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.

相机模型由 Params::camera_model 选择：
  * pinhole（默认）：沿用原实现，cv::aruco 直接用真实内参和 1x5 radtan 畸变；
  * fisheye：Kannala-Brandt（cv::fisheye，k1..k4）。先按模型有效域整块过滤 marker，
    再把原始鱼眼角点投到与 K 同尺度的虚拟针孔平面，之后用同一套 estimatePoseBoard
    求位姿；调试绘制直接画在原始鱼眼图上。
*/

#ifndef QR_DETECT_HPP
#define QR_DETECT_HPP
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <ros/ros.h>
#include <opencv2/aruco.hpp>
#include "common_lib.h"

class QRDetect 
{
  private:
    double marker_size_, delta_width_qr_center_, delta_height_qr_center_;
    double delta_width_circles_, delta_height_circles_;
    int min_detected_markers_;
    cv::Ptr<cv::aruco::Dictionary> dictionary_;
    bool fisheye_;                  // 相机模型是否为鱼眼
    double fisheye_max_theta_deg_;  // 鱼眼角点可用的离轴角上限

  public:
    ros::Publisher qr_pub_;
    cv::Mat imageCopy_;
    // 针孔：CV_32F 3x3 + 1x5 [k1 k2 p1 p2 k3]；鱼眼：CV_64F 3x3 + 4x1 [k1 k2 k3 k4]
    cv::Mat cameraMatrix_;
    cv::Mat distCoeffs_;

    QRDetect(ros::NodeHandle &nh, Params& params) 
    {
      marker_size_ = params.marker_size;
      delta_width_qr_center_ = params.delta_width_qr_center;
      delta_height_qr_center_ = params.delta_height_qr_center;
      delta_width_circles_ = params.delta_width_circles;
      delta_height_circles_ = params.delta_height_circles;
      min_detected_markers_ = params.min_detected_markers;
      fisheye_ = isFisheyeCameraModel(params);
      fisheye_max_theta_deg_ = params.fisheye_max_theta_deg;

      if (fisheye_)
      {
        // 原始鱼眼内参：cv::fisheye 要求畸变恰好是 4 个系数 [k1 k2 k3 k4]
        cameraMatrix_ = (cv::Mat_<double>(3, 3) << params.fx, 0, params.cx,
                                                   0, params.fy, params.cy,
                                                   0,         0,        1);
        distCoeffs_ = (cv::Mat_<double>(4, 1) << params.k1, params.k2,
                                                 params.k3, params.k4);
        ROS_INFO("[QRDetect] fisheye (Kannala-Brandt) model: fx=%.4f fy=%.4f cx=%.4f cy=%.4f "
                 "D=[%+.8f %+.8f %+.8f %+.8f] max_theta=%.1f deg",
                 params.fx, params.fy, params.cx, params.cy,
                 params.k1, params.k2, params.k3, params.k4,
                 fisheye_max_theta_deg_);
      }
      else
      {
        // 针孔分支保持原样：1x5 的最后一项是 radtan 的 k3，不是占位符
        cameraMatrix_ = (cv::Mat_<float>(3, 3) << params.fx, 0, params.cx,
                                                  0, params.fy, params.cy,
                                                  0,         0,        1);
        distCoeffs_ = (cv::Mat_<float>(1, 5) << params.k1, params.k2, params.p1,
                                                params.p2, 0);
      }

      // Initialize QR dictionary
      dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

      qr_pub_ = nh.advertise<sensor_msgs::PointCloud2>("qr_cloud", 1);
    }

    Point2f projectPointDist(cv::Point3f pt_cv, const Mat intrinsics, const Mat distCoeffs) const
    {
      // Project a 3D point taking into account distortion
      vector<Point3f> input{pt_cv};
      vector<Point2f> projectedPoints;
      projectedPoints.resize(1);  // TODO: Do it batched? (cv::circle is not batched anyway)
      projectPoints(input, Mat::zeros(3, 1, CV_64FC1), Mat::zeros(3, 1, CV_64FC1),
      intrinsics, distCoeffs, projectedPoints);
      return projectedPoints[0];
    }

    // 按相机模型把 detectMarkers 的输出整理成位姿估计可用的观测：
    //   * pinhole：直接引用原始角点与真实内参（不做任何数值转换）；
    //   * fisheye：按模型有效域整块过滤 marker，并把角点投到与 K 同尺度的虚拟针孔平面，
    //     此时畸变为零。
    void preparePoseObservations(const std::vector<int>& ids,
                                 const std::vector<std::vector<cv::Point2f>>& corners,
                                 std::vector<int>& ids_used,
                                 std::vector<std::vector<cv::Point2f>>& corners_used,
                                 cv::Mat& intrinsics_used,
                                 cv::Mat& distCoeffs_used) const
    {
      if (!fisheye_)
      {
        ids_used = ids;
        corners_used = corners;
        intrinsics_used = cameraMatrix_;
        distCoeffs_used = distCoeffs_;
        return;
      }

      std::vector<std::vector<cv::Point2f>> corners_virt;
      std::vector<char> marker_valid;
      undistortFisheyeObservations(corners, cameraMatrix_, distCoeffs_,
                                   fisheye_max_theta_deg_, corners_virt, marker_valid);

      std::vector<int> dropped_ids;
      ids_used.clear();
      corners_used.clear();
      for (size_t i = 0; i < marker_valid.size(); ++i)
      {
        if (!marker_valid[i])
        {
          if (i < ids.size()) dropped_ids.push_back(ids[i]);
          continue;
        }
        if (i < ids.size()) ids_used.push_back(ids[i]);
        corners_used.push_back(corners_virt[i]);
      }

      if (!dropped_ids.empty())
      {
        std::ostringstream oss;
        for (size_t i = 0; i < dropped_ids.size(); ++i)
        {
          if (i) oss << ", ";
          oss << dropped_ids[i];
        }
        ROS_WARN("[QRDetect] Dropped %zu marker(s) outside the fisheye model domain "
                 "(> %.1f deg): id %s", dropped_ids.size(), fisheye_max_theta_deg_,
                 oss.str().c_str());
      }

      // 角点已去畸变到虚拟针孔平面，位姿求解用零畸变
      intrinsics_used = cameraMatrix_;
      distCoeffs_used = cv::Mat::zeros(4, 1, CV_64F);
    }

    // 把一个相机系下的 3D 点投到图像像素（模型相关）
    cv::Point2f projectPointCameraFrame(const cv::Point3f& pt_cam) const
    {
      if (fisheye_)
      {
        vector<Point3f> input{pt_cam};
        vector<Point2f> projectedPoints;
        projectedPoints.resize(1);
        cv::fisheye::projectPoints(input, projectedPoints,
                                   cv::Mat::zeros(3, 1, CV_64F),
                                   cv::Mat::zeros(3, 1, CV_64F),
                                   cameraMatrix_, distCoeffs_);
        return projectedPoints[0];
      }
      return projectPointDist(pt_cam, cameraMatrix_, distCoeffs_);
    }

    // 在图像上画出位姿坐标轴（模型相关；鱼眼下 cv::aruco::drawAxis 不适用）
    void drawAxisCameraFrame(const cv::Vec3d& rvec, const cv::Vec3d& tvec, float length) const
    {
      if (!fisheye_)
      {
        cv::aruco::drawAxis(imageCopy_, cameraMatrix_, distCoeffs_, rvec, tvec, length);
        return;
      }

      cv::Mat R;
      cv::Rodrigues(rvec, R);
      const cv::Mat t = (cv::Mat_<double>(3, 1) << tvec[0], tvec[1], tvec[2]);
      const cv::Point3f origin(0.f, 0.f, 0.f);
      const cv::Point3f axis_x(length, 0.f, 0.f);
      const cv::Point3f axis_y(0.f, length, 0.f);
      const cv::Point3f axis_z(0.f, 0.f, length);

      auto toCameraFrame = [&](const cv::Point3f& pt) {
        cv::Mat p = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
        cv::Mat q = R * p + t;
        return cv::Point3f(static_cast<float>(q.at<double>(0)),
                           static_cast<float>(q.at<double>(1)),
                           static_cast<float>(q.at<double>(2)));
      };

      const cv::Point2f uv_origin = projectPointCameraFrame(toCameraFrame(origin));
      const cv::Point2f uv_x = projectPointCameraFrame(toCameraFrame(axis_x));
      const cv::Point2f uv_y = projectPointCameraFrame(toCameraFrame(axis_y));
      const cv::Point2f uv_z = projectPointCameraFrame(toCameraFrame(axis_z));

      line(imageCopy_, uv_origin, uv_x, cv::Scalar(0, 0, 255), 2);
      line(imageCopy_, uv_origin, uv_y, cv::Scalar(0, 255, 0), 2);
      line(imageCopy_, uv_origin, uv_z, cv::Scalar(255, 0, 0), 2);
    }

    void comb(int N, int K, std::vector<std::vector<int>> &groups) {
      int upper_factorial = 1;
      int lower_factorial = 1;

      for (int i = 0; i < K; i++) {
        upper_factorial *= (N - i);
        lower_factorial *= (K - i);
      }
      int n_permutations = upper_factorial / lower_factorial;

      if (DEBUG)
        cout << N << " centers found. Iterating over " << n_permutations
            << " possible sets of candidates" << endl;

      std::string bitmask(K, 1);  // K leading 1's
      bitmask.resize(N, 0);       // N-K trailing 0's

      // print integers and permute bitmask
      do {
        std::vector<int> group;
        for (int i = 0; i < N; ++i)  // [0..N-1] integers
        {
          if (bitmask[i]) {
            group.push_back(i);
          }
        }
        groups.push_back(group);
      } while (std::prev_permutation(bitmask.begin(), bitmask.end()));

      assert(groups.size() == n_permutations);
    }

    void detect_qr(cv::Mat &image, pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud) 
    {      
      image.copyTo(imageCopy_);

      // Create vector of markers corners. 4 markers * 4 corners
      // Markers order:
      // 0-------1
      // |       |
      // |   C   |
      // |       |
      // 3-------2

      // WARNING: IDs are in different order:
      // Marker 0 -> aRuCo ID: 1
      // Marker 1 -> aRuCo ID: 2
      // Marker 2 -> aRuCo ID: 4
      // Marker 3 -> aRuCo ID: 3

      std::vector<std::vector<cv::Point3f>> boardCorners;
      std::vector<cv::Point3f> boardCircleCenters;
      std::vector<int> boardIds;  // IDs order as explained above
      buildTargetBoardGeometry(marker_size_, delta_width_qr_center_,
                               delta_height_qr_center_, delta_width_circles_,
                               delta_height_circles_, boardCorners,
                               boardCircleCenters, boardIds);
      cv::Ptr<cv::aruco::Board> board =
          cv::aruco::Board::create(boardCorners, dictionary_, boardIds);

      cv::Ptr<cv::aruco::DetectorParameters> parameters =
          cv::aruco::DetectorParameters::create();
      // set tp use corner refinement for accuracy, values obtained
      // for pixel coordinates are more accurate than the neaterst pixel

    #if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
      parameters->doCornerRefinement = true;
    #else
      parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    #endif

      // Detect markers
      std::vector<int> ids;
      std::vector<std::vector<cv::Point2f>> corners;
      cv::aruco::detectMarkers(image, dictionary_, corners, ids, parameters);

      // Draw detections if at least one marker detected
      if (ids.size() > 0) cv::aruco::drawDetectedMarkers(imageCopy_, corners, ids);

      // 按相机模型准备位姿估计用的观测：鱼眼需要先按模型有效域过滤再投到虚拟针孔平面
      std::vector<int> ids_used;
      std::vector<std::vector<cv::Point2f>> corners_used;
      cv::Mat intrinsics_used, distCoeffs_used;
      preparePoseObservations(ids, corners, ids_used, corners_used,
                              intrinsics_used, distCoeffs_used);

      cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);  // Vectors to store initial guess
      
      // cout << "min_detected_markers_: " << min_detected_markers_ << std::endl;

      // 用过滤后的 marker（鱼眼下可能少于 detectMarkers 的原始结果）估计位姿
      if (ids_used.size() >= static_cast<size_t>(min_detected_markers_) &&
          ids_used.size() <= TARGET_NUM_MARKERS)
      {
        // pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr candidates_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // 估计标定板位姿（先用各 marker 位姿的平均值作为初值，避开平面靶标的镜像解）
        std::vector<cv::Vec3d> marker_rvecs, marker_tvecs;
        if (!estimateBoardPoseFromMarkers(board, corners_used, ids_used, marker_size_,
                                          intrinsics_used, distCoeffs_used,
                                          marker_rvecs, marker_tvecs, rvec, tvec))
        {
          ROS_WARN("[Mono] Unable to estimate the calibration board pose from %zu "
                   "detected marker(s)", ids_used.size());
          return;
        }

        // Draw markers' axis in color image (Debug purposes)
        for (size_t i = 0; i < marker_rvecs.size(); ++i) {
          drawAxisCameraFrame(marker_rvecs[i], marker_tvecs[i], 0.1);
        }

        // cout << "board: " <<  tvec[0] << ", "<< tvec[1] << ", " << tvec[2] << std::endl;

        drawAxisCameraFrame(rvec, tvec, 0.2);

        // Build transformation matrix to calibration target axis
        cv::Mat R(3, 3, cv::DataType<float>::type);
        cv::Rodrigues(rvec, R);

        cv::Mat t = cv::Mat::zeros(3, 1, CV_32F);
        t.at<float>(0) = tvec[0];
        t.at<float>(1) = tvec[1];
        t.at<float>(2) = tvec[2];

        cv::Mat board_transform = cv::Mat::eye(3, 4, CV_32F);
        R.copyTo(board_transform.rowRange(0, 3).colRange(0, 3));
        t.copyTo(board_transform.rowRange(0, 3).col(3));

        // Compute coordintates of circle centers
        for (int i = 0; i < boardCircleCenters.size(); ++i) {
          cv::Mat mat = cv::Mat::zeros(4, 1, CV_32F);
          mat.at<float>(0, 0) = boardCircleCenters[i].x;
          mat.at<float>(1, 0) = boardCircleCenters[i].y;
          mat.at<float>(2, 0) = boardCircleCenters[i].z;
          mat.at<float>(3, 0) = 1.0;

          // Transform center to target coords
          cv::Mat mat_qr = board_transform * mat;
          cv::Point3f center3d;
          center3d.x = mat_qr.at<float>(0, 0);
          center3d.y = mat_qr.at<float>(1, 0);
          center3d.z = mat_qr.at<float>(2, 0);

          // Draw center (DEBUG)
          cv::Point2f uv;
          uv = projectPointCameraFrame(center3d);
          circle(imageCopy_, uv, 5, Scalar(0, 255, 0), -1);

          // Add center to list
          pcl::PointXYZ qr_center;
          qr_center.x = center3d.x;
          qr_center.y = center3d.y;
          qr_center.z = center3d.z;
          candidates_cloud->push_back(qr_center);
        }

        /**
          NOTE: This is included here in the same way as the rest of the modalities
        to avoid obvious misdetections, which sometimes happened in our experiments.
        In this modality, it should be impossible to have more than a set of
        candidates, but we keep the ability of handling different combinations for
        eventual future extensions.

          Geometric consistency check
          At this point, circles' center candidates have been computed
        (found_centers). Now we need to select the set of 4 candidates that best fit
        the calibration target geometry. To that end, the following steps are
        followed: 1) Create a cloud with 4 points representing the exact geometry of
        the calibration target 2) For each possible set of 4 points: compute
        similarity score 3) Rotate back the candidates with the highest score to
        their original position in the cloud, and add them to cumulative cloud
        **/
        std::vector<std::vector<int>> groups;
        comb(candidates_cloud->size(), TARGET_NUM_CIRCLES, groups);
        double groups_scores[groups.size()];  // -1: invalid; 0-1 normalized score
        // groups.size() 1

        for (int i = 0; i < groups.size(); ++i) 
        {
          std::vector<pcl::PointXYZ> candidates;
          // Build candidates set
          for (int j = 0; j < groups[i].size(); ++j) {
            pcl::PointXYZ center;
            center.x = candidates_cloud->at(groups[i][j]).x;
            center.y = candidates_cloud->at(groups[i][j]).y;
            center.z = candidates_cloud->at(groups[i][j]).z;
            candidates.push_back(center);
          }

          // Compute candidates score
          Square square_candidate(candidates, delta_width_circles_,
                                  delta_height_circles_);
          groups_scores[i] = square_candidate.is_valid()
                                ? 1.0
                                : -1;  // -1 when it's not valid, 1 otherwise
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (int i = 0; i < groups.size(); ++i) 
        {
          if (best_candidate_score == 1 && groups_scores[i] == 1) {
            // Exit 4: Several candidates fit target's geometry
            ROS_ERROR(
                "[Mono] More than one set of candidates fit target's geometry. "
                "Please, make sure your parameters are well set. Exiting callback");
            return;
          }
          if (groups_scores[i] > best_candidate_score) {
            best_candidate_score = groups_scores[i];
            best_candidate_idx = i;
          }
        }

        if (best_candidate_idx == -1) 
        {
          // Exit: No candidates fit target's geometry
          ROS_WARN(
              "[Mono] Unable to find a candidate set that matches target's "
              "geometry");
          return;
        }

        for (int j = 0; j < groups[best_candidate_idx].size(); ++j) 
        {
          centers_cloud->push_back(candidates_cloud->at(groups[best_candidate_idx][j]));
        }

        if (DEBUG) 
        {  // Draw centers
          for (int i = 0; i < centers_cloud->size(); i++) {
            cv::Point3f pt_circle1(centers_cloud->at(i).x, centers_cloud->at(i).y,centers_cloud->at(i).z);
            cv::Point2f uv_circle1;
            uv_circle1 = projectPointCameraFrame(pt_circle1);
            circle(imageCopy_, uv_circle1, 2, Scalar(255, 0, 255), -1);
          }
        }

        // Publish pointcloud messages
      } 
      else 
      {
        // Markers found != TARGET_NUM_CIRCLES
        ROS_WARN("%lu marker(s) found, %d expected. Skipping frame...", ids_used.size(),
                TARGET_NUM_MARKERS);
      }
    }
};
typedef std::shared_ptr<QRDetect> QRDetectPtr;

#endif