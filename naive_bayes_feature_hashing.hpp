#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesFeatureHashing : public BaseClf<NaiveBayesFeatureHashing> {
    int log_num_buckets_;

    // vector for ngrams in spam mails
    std::vector<int> spam_buckets_;

    // vector for ngrams in ham mails
    std::vector<int> ham_buckets_;

    // counters to count number of spam and ham mails
    int spam_count_;
    int ham_count_;

    int seed_;

public:
    NaiveBayesFeatureHashing(int log_num_buckets, double threshold)
        : log_num_buckets_(log_num_buckets)
        , seed_(0x249cd)
    {
        // initialize features with 1 to avoid zero counts
        spam_buckets_ = std::vector(pow(2, log_num_buckets_), 1);
        ham_buckets_ = std::vector(pow(2,log_num_buckets_),1);
        spam_count_ = 1;
        ham_count_ = 1;

        this->threshold = threshold;
    }

    void update_(const Email& email)
    {
        // check ham or spam and increment count
        if (email.is_spam()) {
            ++spam_count_;
            update_buckets_(email,true);
        } else {
            ++ham_count_;
            update_buckets_(email,false);
        }
    }

    void update_buckets_(const Email& email, bool spam)
    {
        EmailIter emailIter_ = EmailIter(email,ngram_k);

        while (!emailIter_.is_done()) {
            std::string_view ngram_ = emailIter_.next();
            size_t hash_value = get_bucket(ngram_);
            if (spam) {
                ++spam_buckets_[hash_value];
            }
            else {
                ++ham_buckets_[hash_value];
            }
        }
    }

    double predict_(const Email& email) const
    {
        /*
        To calculate the prediction, the formula on slide 10 of the lecture slides on scaling learning is used.
        The result of this formula will be positive if the email is spam and negative if the email is ham, so threshold that should be used is 0.
        The result will be positive if the probability on the email being spam is higher than the probability of the email being ham.
        We can use this formula because we only care about the classification, so about the result being positive or negative.
             prob(Spam email)         P(S)         P(X1|S)         P(X2|S)               P(Xn|S)
        log ------------------ = log ------ + log --------- + log --------- + ... + log ---------
             prob(Ham email)          P(H)         P(X1|H)         P(X2|H)               P(Xn|H)
        */

        double result_;
        EmailIter emailIter_ = EmailIter(email,ngram_k);

        // probability of email being spam
        double p_s_ = (1.0 * spam_count_) / (spam_count_ + ham_count_);
        // probability of email being ham
        double p_h_ = (1.0 * ham_count_) / (spam_count_ + ham_count_);

        // first part of the formula
        result_ = log(p_s_ / p_h_);

        while (!emailIter_.is_done()) {
            std::string_view ngram_ = emailIter_.next();
            size_t hash_value_ = get_bucket(ngram_);

            // get count of ngram in spam mails from counts vector
            int ngram_spam_count_ = spam_buckets_[hash_value_];
            // probability of ngram present in the email given that the email is spam
            double p_ngram_s_ = (1.0 * ngram_spam_count_) / spam_count_;

            // get count of ngram in ham mails from counts vector
            int ngram_ham_count_ = ham_buckets_[hash_value_];
            // probability of ngram present in the email given that the email is ham
            double p_ngram_h_ = (1.0 * ngram_ham_count_) / ham_count_;

            // update the formula for each ngram
            result_ = result_ + log(p_ngram_s_ / p_ngram_h_);
        }
        return result_;
    }

    void print_weights() const
    {
        size_t n = (1 << log_num_buckets_);
        for (size_t i = 0; i < n; ++i)
        {
            std::cout << "w" << i << " " << spam_buckets_[i] << ", " << spam_buckets_[n + i] << std::endl;
        }
    }

private:
    size_t get_bucket(std::string_view ngram) const
    { return get_bucket(hash(ngram, seed_)); }

    size_t get_bucket(size_t hash) const {
        // desired range equals 2^log_num_buckets
        // modulo division to convert to desired range
        size_t bucket_ = hash % (size_t)pow(2,log_num_buckets_);
        return bucket_;
    }
};

} // namespace bdap
