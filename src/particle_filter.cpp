/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 * Modified on: Oct 02, 2020
 * Author: Vinayak Kamath
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;

using std::normal_distribution;
static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   */
  num_particles = 50;  // TODO: Set the number of particles

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Normal distributions for the respective means and standard deviations.
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  /**
  * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  for (int i = 0; i < num_particles; i++) {
      Particle CurrentParticle;
      CurrentParticle.id = i;
      CurrentParticle.x = dist_x(gen);
      CurrentParticle.y = dist_y(gen);
      CurrentParticle.theta = dist_theta(gen);
      CurrentParticle.weight = 1.0;

      particles.push_back(CurrentParticle); //order it into the particles vector
      weights.push_back(CurrentParticle.weight); //order it into the weights vector
  }
  is_initialized = true; //flagged indication that initialization has concluded.
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

    double std_x = std_pos[0];
    double std_y = std_pos[1];
    double std_theta = std_pos[2];

    // Get normal distributions for noise
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for (int i = 0; i < num_particles; i++) {
        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        // Step towards Prediction
        if (fabs(yaw_rate) > 0.0000001) {
            x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
            y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
            theta += yaw_rate * delta_t;
        }

        else { //avoid small yaw_rates and have issues with division by zero errors
            x += velocity * delta_t * cos(theta);
            y += velocity * delta_t * sin(theta);
        }

        // Adding noise usind default_random_engine
        particles[i].x = x + dist_x(gen);
        particles[i].y = y + dist_y(gen);
        particles[i].theta = theta + dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
    // Go through all observations and associate the closest landmark accord calc. distance to each obs.
    for (unsigned int i = 0; i < observations.size(); i++) {
        double obs_x = observations[i].x;
        double obs_y = observations[i].y;
        double distance;
        double min_dist = std::numeric_limits<double>::infinity();
        int nearestId = -1;

        for (unsigned int j = 0; j < predicted.size(); j++) {
            double preds_x = predicted[j].x;
            double preds_y = predicted[j].y;
            distance = dist(obs_x, obs_y, preds_x, preds_y);
            if (distance < min_dist) {
                min_dist = distance;
                nearestId = predicted[j].id;
            }
        }
        observations[i].id = nearestId;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    double std_landmark_x = std_landmark[0];
    double std_landmark_y = std_landmark[1];

    for (int i = 0; i < num_particles; i++) {
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;
        vector<LandmarkObs> landmarkWithinSensorRange;

        for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
            double map_x = map_landmarks.landmark_list[j].x_f;
            double map_y = map_landmarks.landmark_list[j].y_f;
            int landmark_id = map_landmarks.landmark_list[j].id_i;
            double dist_particle2landmark = dist(particle_x, particle_y, map_x, map_y);

            if (dist_particle2landmark<=sensor_range) {
                // push measurement to vector of immediate landmarks
                landmarkWithinSensorRange.push_back(LandmarkObs{ landmark_id,map_x,map_y });
            }
        }
        // Transform observations from Vehicle to map coordinate system
        vector<LandmarkObs> transformedObservations;
        for (unsigned int o = 0; o < observations.size(); o++) {
            int observation_id = observations[o].id;
            double observation_x = observations[o].x;
            double observation_y = observations[o].y;

            double transformed_x = particle_x + cos(particle_theta) * observation_x - sin(particle_theta) * observation_y;
            double transformed_y = particle_y + sin(particle_theta) * observation_x + cos(particle_theta) * observation_y;
            transformedObservations.push_back(LandmarkObs{ observation_id,transformed_x,transformed_y });
        }

        //Associate transformed observations to nearest landmark
        dataAssociation(landmarkWithinSensorRange, transformedObservations);

        // Update the weights of each particle using a mult-variate Gaussian Dist.
        particles[i].weight = 1.0;
        for (unsigned int t = 0; t < transformedObservations.size(); t++) {
            double x = transformedObservations[t].x;
            double y = transformedObservations[t].y;
            int associatedLandmarkId = transformedObservations[t].id;

            //get the x and y coordinates of the nearest landmark
            double landmark_x, landmark_y;
            for (unsigned int l = 0; l < landmarkWithinSensorRange.size(); ++l) {
                if (landmarkWithinSensorRange[l].id == associatedLandmarkId) {
                    landmark_x = landmarkWithinSensorRange[l].x;
                    landmark_y = landmarkWithinSensorRange[l].y;
                    break;
                }
            }

            // Calculate weights as probabilities
            double weight_obs = multiv_prob(std_landmark_x, std_landmark_y, x, y, landmark_x, landmark_y);
            particles[i].weight* weight_obs; // product of all observation weights
            weights[i] = particles[i].weight;
        }
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
    // Take information from resampling wheel implementation code
    vector<Particle> sampledParticles;

    double beta = 0.0;
    double MaxWeight = *std::max_element(weights.begin(), weights.end());
    std::uniform_real_distribution<double> distribution(0.0, MaxWeight);
    std::uniform_int_distribution<int> int_distribution(0, num_particles - 1);

    int index = int_distribution(gen);
    for (int i = 0; i < num_particles; i++) {
        beta += distribution(gen) * 2.0;
        while (weights[index] < beta) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        sampledParticles.push_back(particles[index]);

    }
    particles = sampledParticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}