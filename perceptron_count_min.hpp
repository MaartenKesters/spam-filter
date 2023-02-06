#pragma once

#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class PerceptronCountMin : public BaseClf<PerceptronCountMin> {
    int log_num_buckets_;
    int num_hashes_;
    double learning_rate_;
    double bias_;

    // vectors for weight of the words hashed with different hash functions placed in one long vector behind each other
    std::vector<double> weights_;
    std::vector<double> input_;

    int seed_;

public:
    PerceptronCountMin(int num_hashes, int log_num_buckets, double learning_rate)
        : num_hashes_(num_hashes)
        , log_num_buckets_(log_num_buckets)
        , learning_rate_(learning_rate)
        , bias_(0.0)
        , seed_(0x9748cd)
    {
        // set all weights and input to zero
        weights_ = std::vector((long) pow(2, log_num_buckets_) * num_hashes, 0.0);
        input_ = std::vector((long) pow(2, log_num_buckets_) * num_hashes, 0.0);
    }

    void update_(const Email& email)
    {
        // get predicted value of email
        double pred_ = predict_(email);

        // normalize real output to (-1,1)
        double real_ = (2.0 * ((double) email.is_spam()) / 1.0) - 1.0;

        // calculate the error
        double error_ = real_ - pred_;

        // input feature vector for the different hashes placed behind each other in the same vector
        input_ = std::vector((long) pow(2, log_num_buckets_) * num_hashes_, 0.0);
        get_feature_vector_(email, input_);

        // update bias term
        bias_ = bias_ + (learning_rate_ * error_);

        // update weights
        for (int i = 0; i < weights_.size(); ++i) {
            // update the weight value for the correct hash function
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
        // input feature vector for the different hashes placed behind each other in the same vector (would be better if we use the global input_ vector here to avoid initializing a new vector, but this is not possible because predict_ function is made const)
        std::vector<double> input_pred_ = std::vector((long) pow(2, log_num_buckets_) * num_hashes_, 0.0);
        get_feature_vector_(email, input_pred_);

        // calculate the output of the perceptron
        double result_ = 0.0;

        // add bias term
        result_ = result_ + bias_;

        // combine weights with input features
        EmailIter emailIter_ = EmailIter(email,ngram_k);
        while (!emailIter_.is_done()) {
            std::string_view ngram_ = emailIter_.next();

            // compute the different hashes for the ngram and calculate the median weight
            std::vector<double> ngram_weights_;
            // compute the average input value
            double input_sum_ = 0.0;
            for (int i = 0; i < num_hashes_; ++i) {
                size_t hash_value_ = get_bucket(ngram_, i);
                double w_ = weights_[(long) (pow(2, log_num_buckets_) * i) + hash_value_];
                double input_val_ = input_pred_[(long) (pow(2, log_num_buckets_) * i) + hash_value_];

                ngram_weights_.push_back(w_);
                input_sum_ = input_sum_ + input_val_;

                // set input value to 0, so we only use each weight once in combination with its input
                input_pred_[(long) (pow(2, log_num_buckets_) * i) + hash_value_] = 0;
            }
            // average input
            double avg_input_ = input_sum_ / (double) num_hashes_;
            // sort the weights
            sort(ngram_weights_.begin(), ngram_weights_.end());
            // number of hashes is even
            if (num_hashes_ % 2 == 0) {
                double median = (ngram_weights_[(num_hashes_ - 1) / 2] + ngram_weights_[num_hashes_ / 2]) / 2.0;
                result_ = result_ + (median * avg_input_);
            }
            // number of hashes is odd
            else {
                double median = ngram_weights_[num_hashes_ / 2];
                result_ = result_ + (median * avg_input_);
            }
        }

        return result_;
    }

    void get_feature_vector_(const Email& email, std::vector<double>& in_) const
    {
        for (int i = 0; i < num_hashes_; ++i) {
            EmailIter emailIter_ = EmailIter(email,ngram_k);
            while (!emailIter_.is_done()) {
                std::string_view ngram_ = emailIter_.next();
                size_t hash_value_ = get_bucket(ngram_, i);
                in_[(long) (pow(2, log_num_buckets_) * i) + hash_value_] = in_[(long) (pow(2, log_num_buckets_) * i) + hash_value_] + 1.0;
            }
            // max value of input feature vector
            double max_value_ = *max_element(std::begin(in_) + ((long)pow(2, log_num_buckets_) * i), std::begin(in_) + (long)(pow(2, log_num_buckets_) * (i + 1)) - 1);

            // normalize the values of the input feature vector so input is between 0 and 1, where max_value equals 1
            transform(std::begin(in_) + ((long)pow(2, log_num_buckets_) * i), std::begin(in_) + (long)(pow(2, log_num_buckets_) * (i + 1)), std::begin(in_) + (long)(pow(2, log_num_buckets_) * i), [max_value_](double &val){ return val/max_value_; });
        }
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
    size_t get_bucket(std::string_view ngram, int hash_num) const
    { return get_bucket(hash(ngram, seed_ + hash_num)); } // increase the seed to get different hashes

    size_t get_bucket(size_t hash) const {
        // desired range equals 2^log_num_buckets
        // modulo division to convert to desired range
        size_t bucket_ = hash % (size_t)(pow(2,log_num_buckets_));
        return bucket_;
    }
};

} // namespace bdap
