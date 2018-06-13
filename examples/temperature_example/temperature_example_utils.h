/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_TEMPERATURE_EXAMPLE_UTILS_H
#define ALBATROSS_TEMPERATURE_EXAMPLE_UTILS_H

#include "csv.h"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "core/model.h"
#include "covariance_functions/covariance_functions.h"
#include "models/gp.h"

namespace albatross {

/*
 * Holds the information about a single station which
 * is used as the FeatureType for our Gaussian process.
 */
struct Station {
  int id;
  double lat;
  double lon;
  double height;
  Eigen::Vector3d ecef;

  bool operator==(const Station &rhs) const { return (ecef == rhs.ecef); }

  template <typename Archive> void serialize(Archive &archive) {
    archive(id, lat, lon, height, ecef);
  }
};

/*
 * Provides an interface which maps the Station.ecef
 * field to an arbitrary DistanceMetric defined on Eigen
 * vectors.
 */
template <typename DistanceMetricType>
class StationDistance : public DistanceMetricType {
public:
  StationDistance(){};

  std::string get_name() const {
    std::ostringstream oss;
    oss << "station_" << DistanceMetricType::get_name();
    return oss.str();
  };

  ~StationDistance(){};

  double operator()(const Station &x, const Station &y) const {
    return DistanceMetricType::operator()(x.ecef, y.ecef);
  };
};

class ElevationScalingFunction : public albatross::ScalingFunction {
public:
  ElevationScalingFunction(double center = 1000., double factor = 3.5 / 300) {
    this->params_["elevation_scaling_center"] = center;
    this->params_["elevation_scaling_factor"] = factor;
  };

  std::string get_name() const { return "elevation_scaled"; }

  double operator()(const Station &x) const {
    // This is the negative orientation rectifier function which
    // allows lower elevations to have a higher variance.
    double center = this->params_.at("elevation_scaling_center");
    return 1. +
           this->params_.at("elevation_scaling_factor") *
               fmax(0., (center - x.height));
  }
};

albatross::RegressionDataset<Station>
read_temperature_csv_input(std::string file_path, int thin = 5) {
  std::vector<Station> features;
  std::vector<double> targets;

  io::CSVReader<8> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "STATION", "LAT", "LON",
                      "ELEV(M)", "X", "Y", "Z", "TEMP");

  bool more_to_parse = true;
  int count = 0;
  while (more_to_parse) {
    double temperature;
    Station station;
    more_to_parse = file_in.read_row(
        station.id, station.lat, station.lon, station.height, station.ecef[0],
        station.ecef[1], station.ecef[2], temperature);
    if (more_to_parse && count % thin == 0) {
      features.push_back(station);
      targets.push_back(temperature);
    }
    count++;
  }
  Eigen::Map<Eigen::VectorXd> eigen_targets(&targets[0],
                                            static_cast<int>(targets.size()));
  return albatross::RegressionDataset<Station>(features, eigen_targets);
}

inline bool file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

void write_predictions(const std::string output_path,
                       const std::vector<Station> features,
                       const albatross::RegressionModel<Station> &model) {

  std::ofstream ostream;
  ostream.open(output_path);
  ostream << "STATION,LAT,LON,ELEV(M),X,Y,Z,TEMP,VARIANCE" << std::endl;

  std::size_t n = features.size();
  std::size_t count = 0;
  for (const auto &f : features) {
    ostream << std::to_string(f.id);
    ostream << ", " << std::to_string(f.lat);
    ostream << ", " << std::to_string(f.lon);
    ostream << ", " << std::to_string(f.height);
    ostream << ", " << std::to_string(f.ecef[0]);
    ostream << ", " << std::to_string(f.ecef[1]);
    ostream << ", " << std::to_string(f.ecef[2]);

    std::vector<Station> one_feature = {f};
    const auto pred = model.predict(one_feature);

    ostream << ", " << std::to_string(pred.mean[0]);
    ostream << ", " << std::to_string(std::sqrt(pred.covariance(0, 0)));
    ostream << std::endl;
    if (count % 1000 == 0) {
      std::cout << count + 1 << "/" << n << std::endl;
    }
    count++;
  }
}

} // namespace albatross
#endif
