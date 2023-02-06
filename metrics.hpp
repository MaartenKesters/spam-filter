#include "email.hpp"

namespace bdap {

    struct Accuracy {
        int n = 0;
        int correct = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);
            ++n;
            correct += static_cast<int>(lab == pred);
        }

        double get_accuracy() const { return static_cast<double>(correct) / n; }
        double get_error() const { return 1.0 - get_accuracy(); }

        double get_score() const { return get_accuracy(); }
    };

    struct ConfusionMatrix {
        int tp = 0;
        int fp = 0;
        int tn = 0;
        int fn = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            bool pred = clf.classify(pr);

            if (lab == 1) {
                if (pred == 1) {
                    ++tp;
                }
                else {
                    ++fn;
                }
            }
            else {
                if (pred == 1) {
                    ++fp;
                }
                else {
                    ++tn;
                }
            }
        }

        double get_accuracy() const { return static_cast<double>(tp+tn) / (tp+tn+fp+fn); }
        double get_recall() const { return static_cast<double>(tp) / (tp+fn); }
        double get_precision() const { return static_cast<double>(tp) / (tp+fp); }
        double get_f1score() const { return static_cast<double>(1+0.5)*get_precision()*get_recall() / ((0.5*get_recall())+get_precision()); } // give more weight to precision

        double get_score() const { return get_precision(); }
    };

    struct LogLoss {
        double logLoss_sum_ = 0;
        int count_ = 0;

        template <typename Clf>
        void evaluate(const Clf& clf, const std::vector<Email>& emails)
        {
            for (const Email& email : emails)
                evaluate(clf, email);
        }

        template <typename Clf>
        void evaluate(const Clf& clf, const Email& email) {
            // calculate the logLoss for each email in the window and add it to the total sum
            bool lab = email.is_spam();
            double pr = clf.predict(email);
            double pr_norm = (pr - (-1)) / 2;
            // cheating to avoid getting nan as result when pr_norm is 1.0 or 0.0
            if (pr_norm > 1.0)
                pr_norm = 0.999;
            if (pr_norm < 0.0)
                pr_norm = 0.001;
            double result_ = -(lab * log(pr_norm) + (1 - lab) * log(1 - pr_norm));
            logLoss_sum_ = logLoss_sum_ + result_;
            count_ = count_ + 1;
        }

        // return the average logLoss over all the emails
        double get_score() const { return logLoss_sum_ / count_; }
    };

} // namespace bdap
