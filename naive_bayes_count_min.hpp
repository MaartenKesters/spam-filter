#pragma once

#include <cmath>
#include <iostream>
#include <string_view>
#include <vector>
#include "email.hpp"
#include "base_classifier.hpp"

namespace bdap {

class NaiveBayesCountMin : public BaseClf<NaiveBayesCountMin> {
    int log_num_buckets_;
    int num_hashes_;

    // vectors for ngrams in spam mails hashed with different hash functions placed in one long vector behind each other
    std::vector<int> spam_buckets_;

    // vectors for ngrams in ham mails hashed with different hash functions placed in one long vector behind each other
    std::vector<int> ham_buckets_;

    // counters to count number of spam and ham mails
    int spam_count_;
    int ham_count_;

    int seed_;

public:
    NaiveBayesCountMin(int num_hashes, int log_num_buckets, double threshold)
        : log_num_buckets_(log_num_buckets)
        , num_hashes_(num_hashes)
        , seed_(0x249cd)
    {
        spam_buckets_ = std::vector((long) pow(2, log_num_buckets_) * num_hashes, 1);
        ham_buckets_ = std::vector((long) pow(2, log_num_buckets_) * num_hashes, 1);
        spam_count_ = 1;
        ham_count_ = 1;

        this->threshold = threshold;
    }

    void update_(const Email &email)
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
        // hash the ngram with the different hash functions
        for (int i = 0; i < num_hashes_; ++i) {
            EmailIter emailIter_ = EmailIter(email,ngram_k);
            while (!emailIter_.is_done()) {
                std::string_view ngram_ = emailIter_.next();

                size_t hash_value = get_bucket(ngram_,i);
                if (spam) {
                    ++spam_buckets_[(long) pow(2, log_num_buckets_) * i + hash_value];
                }
                else {
                    ++ham_buckets_[(long) pow(2, log_num_buckets_) * i + hash_value];
                }
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
        double p_s_ = (double) spam_count_ / (spam_count_ + ham_count_);
        // probability of email being ham
        double p_h_ = (double) ham_count_ / (spam_count_ + ham_count_);

        // first part of the formula
        result_ = log(p_s_ / p_h_);

        while (!emailIter_.is_done()) {
            std::string_view ngram_ = emailIter_.next();

            // min count of ngram in spam hash table
            int ngram_min_count_spam_ = 0;
            // min count of ngram in ham hash table
            int ngram_min_count_ham = 0;
            // compute the different hashes for the ngram in the spam and ham hash table and take the min count
            for (int i = 0; i < num_hashes_; ++i) {
                size_t hash_value_ = get_bucket(ngram_,i);

                int ngram_count_spam_ = spam_buckets_[(long) pow(2, log_num_buckets_) * i + hash_value_];
                if (ngram_min_count_spam_ == 0 || ngram_min_count_spam_ > ngram_count_spam_)
                    ngram_min_count_spam_ = ngram_count_spam_;

                int ngram_count_ham_ = ham_buckets_[(long) pow(2, log_num_buckets_) * i + hash_value_];
                if (ngram_min_count_ham == 0 || ngram_min_count_ham > ngram_count_ham_)
                    ngram_min_count_ham = ngram_count_ham_;
            }
            // probability of ngram present in the email given that the email is spam
            double p_ngram_s_ = (1.0 * ngram_min_count_spam_) / spam_count_;
            // probability of ngram present in the email given that the email is ham
            double p_ngram_h_ = (1.0 * ngram_min_count_ham) / ham_count_;

            // update the formula for each ngram
            result_ = result_ + log(p_ngram_s_ / p_ngram_h_);
        }
        return result_;
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
