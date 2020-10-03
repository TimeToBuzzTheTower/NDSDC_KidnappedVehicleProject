/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
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

#include "helper_functions.h"

#include <limits>

using std::string;
using std::vector;

using std::normal_distribution;
static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;  // TODO: Set the number of particles
  double std_x = std[0];  // 0.3
  double std_y = std[1];  // 0.3
  double std_theta = std[2];  // 0.01

  // Generate normal distributions according to respective means and stds
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  
  for(int i=0; i < num_particles; i++){
    Particle pa;  // current particle
    pa.id = i;
    pa.x = dist_x(gen);
    pa.y = dist_y(gen);
    pa.theta = dist_theta(gen);
    pa.weight = 1.0;

    particles.push_back(pa);  // push back to particles vector
    weights.push_back(pa.weight);  // push back to weights vector which is used in the resample() part
  }
  is_initialized = true;  // set flag to indicate that initialization has finished
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  // Generate normal distributions for noise according to mean 0 and respective stds
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for(int i=0; i<num_particles; i++){
    double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;

    // Prediction step
    if (fabs(yaw_rate) > 0.00001) {
      x += velocity/yaw_rate*(sin(theta+yaw_rate*delta_t)-sin(theta));
      y += velocity/yaw_rate*(cos(theta)-cos(theta+yaw_rate*delta_t));
      theta += yaw_rate*delta_t;
    }

    else {  //catch cases where yaw_rate is too small such to avoid division by zero
      x += velocity*delta_t*cos(theta);
      y += velocity*delta_t*sin(theta);
    }

    // Add random gaussian noise
    particles[i].x = x + dist_x(gen);
    particles[i].y = y + dist_y(gen);
    particles[i].theta = theta + dist_theta(gen);
    
	}
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  //Go through all observations; for each obs. associate the closest landmark according to the calculated distance
  for(unsigned int i=0; i<observations.size(); i++){
    double obs_x = observations[i].x;
    double obs_y = observations[i].y;
    double distance;
    double min_dist = std::numeric_limits<double>::infinity();
    int nearestId = -1;

    for(unsigned int j=0; j<predicted.size(); j++){
      double preds_x = predicted[j].x;
      double preds_y = predicted[j].y;
      distance = dist(obs_x, obs_y, preds_x, preds_y);
      if(distance < min_dist){  // update closest landmark id and minimum distance
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

  double std_ldm_x = std_landmark[0];
  double std_ldm_y = std_landmark[1];
  
  // Update weights for each particle
  for(int i=0; i<num_particles; i++){
    double xp = particles[i].x;
    double yp = particles[i].y;
    double thetap = particles[i].theta;
    vector<LandmarkObs> landmarkWithinSensorRange;

    // For each particle, consider only landmarks in vicinity of sensor_range
    for(unsigned int j=0; j<map_landmarks.landmark_list.size();++j){
      double xm = map_landmarks.landmark_list[j].x_f;
      double ym = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i; 
      double dist_particle2lm = dist(xp, yp, xm, ym);

      if(dist_particle2lm<=sensor_range){
        // push measurement to vector of immediate landmarks
        landmarkWithinSensorRange.push_back(LandmarkObs{lm_id, xm, ym});
      }
    }
    // Transform observation coordinates to particles coordinate system
    vector<LandmarkObs> transformedObs;
    for(unsigned int o=0; o<observations.size(); o++){
      int obs_id = observations[o].id;
      double xo = observations[o].x;
      double yo = observations[o].y;

      double xt = xp + cos(thetap) * xo - sin(thetap) * yo;
      double yt = yp + sin(thetap) * xo + cos(thetap) * yo;
      transformedObs.push_back(LandmarkObs{obs_id, xt, yt});
    }
    
    // Associate transformed observations to nearest landmark
    dataAssociation(landmarkWithinSensorRange, transformedObs);


    // Compute weights of particles using multivariate Gaussian
    particles[i].weight = 1.0;
    for(unsigned int t=0; t<transformedObs.size(); t++){
      double x = transformedObs[t].x;
      double y = transformedObs[t].y;
      int associatedLmId = transformedObs[t].id;
      
      // find x and y coordinates of nearest landmark
      double mu_x, mu_y;
      for(unsigned int l=0; l<landmarkWithinSensorRange.size(); ++l){
        if(landmarkWithinSensorRange[l].id == associatedLmId){
          mu_x = landmarkWithinSensorRange[l].x;
          mu_y = landmarkWithinSensorRange[l].y;
          break;
        }
      }

      // calculate weights as probabilities
      double weight_obs = multiv_prob(std_ldm_x, std_ldm_y, x, y, mu_x, mu_y);  // inline function defined in particles.h
      particles[i].weight *= weight_obs;  //calculate final weight of particle (product of all observation weights)
      weights[i] = particles[i].weight;  // use this vector for the resampling step
    }
  }
}

void ParticleFilter::resample() {

  // Resampling according to resampling wheel from the lecture
  vector<Particle> sampledParticles;
  
  double beta = 0.0;
  double mw = *std::max_element(weights.begin(), weights.end());  // max. weight
  std::uniform_real_distribution<double> distribution(0.0, mw);
  std::uniform_int_distribution<int> int_distribution(0, num_particles-1);

  int index = int_distribution(gen);
  for(int i=0; i<num_particles; i++){
    beta += distribution(gen)*2.0;
    while(weights[index] < beta){
      beta -= weights[index];
      index = (index+1) % num_particles;
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