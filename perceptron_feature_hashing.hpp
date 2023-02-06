#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronFeatureHashing : public BaseClf<PerceptronFeatureHashing> {
    int log_num_buckets_;
    double learning_rate_;
    double bias_;
    std::vector<double> weights_;
    std::vector<double> input_;

    int seed_;

public:
    PerceptronFeatureHashing(int log_num_buckets, double learning_rate)
        : log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0x9748cd)
    {
        // set all weights and input to zero
        weights_.resize(1 << log_num_buckets_, 0.0);
        input_.resize(1 << log_num_buckets_, 0.0);
    }

    void update_(const Email& email)
    {
        // get predicted value of email
        double pred_ = predict_(email);

        // normalize real output to (-1,1)
        double real_ = (2.0 * ((double) email.is_spam()) / 1.0) - 1.0;

        // calculate the error
        double error_ = real_ - pred_;

        // input feature vector
        //std::vector<double> input_ = std::vector(pow(2, log_num_buckets_), 0.0);
        input_.resize(1 << log_num_buckets_, 0.0);
        get_feature_vector_(email, input_);

        // update bias term
        bias_ = bias_ + (learning_rate_ * error_);

        // update weights
        for (int i = 0; i < weights_.size(); ++i) {
            weights_[i] = weights_[i] + (learning_rate_ * error_ * input_[i]);
            // limit the weights to (-1, 1) range
            if (weights_[i] > 1.0)
                weights_[i] = 1.0;
            if (weights_[i] < -1.0)
                weights_[i] = -1.0;
        }
    }

    double predict_(const Email& email) const
    {
        // input feature vector (would be better if we use the global input_ vector here to avoid initializing a new vector, but this is not possible because predict_ function is made const)
        std::vector<double> input_pred_ = std::vector(pow(2, log_num_buckets_), 0.0);
        get_feature_vector_(email, input_pred_);

        // calculate the output of the perceptron
        double result_ = 0.0;

        // add bias term
        result_ = result_ + bias_;

        // combine weights with input features
        for (int i = 0; i < weights_.size(); ++i) {
            result_ = result_ + (input_pred_[i] * weights_[i]);
        }

        return result_;
    }

    void get_feature_vector_(const Email& email, std::vector<double>& in_) const {
        EmailIter emailIter_ = EmailIter(email,ngram_k);
        while (!emailIter_.is_done()) {
            std::string_view ngram_ = emailIter_.next();
            size_t hash_value_ = get_bucket(ngram_);
            in_[hash_value_] = in_[hash_value_] + 1.0;
        }

        // max value of input feature vector
        double max_value_ = *max_element(std::begin(in_), std::end(in_));

        // normalize the values of the input feature vector so input is between 0 and 1, where max_value equals 1
        transform(in_.begin(), in_.end(), in_.begin(), [max_value_](double &val){ return val/max_value_; });
    }

    void print_weights() const
    {
        std::cout << "bias " << bias_ << std::endl;
        for (size_t i = 0; i < weights_.size(); ++i)
        {
            std::cout << "w" <<i << " " << weights_[i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const
    {
        // desired range equals 2^log_num_buckets
        // modulo division to convert to desired range
        size_t bucket_ = hash % (size_t)(pow(2,log_num_buckets_));
        return bucket_;
    }
};

} // namespace bdap
