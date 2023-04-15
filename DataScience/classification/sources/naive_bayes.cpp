#include "../headers/naive_bayes.h"

template <typename T>
NaiveBayes<T>::NaiveBayes()
    : numExamples_(0) {}

template <typename T>
NaiveBayes<T>::~NaiveBayes() {}

template <typename T>
void NaiveBayes<T>::addExample(const std::vector<T> &features, const std::string &label)
{
    // Increment the count for the label
    ++labelCounts_[label];

    // Increment the count for each feature and label
    for (size_t i = 0; i < features.size(); ++i)
    {
        const std::string featureLabel = std::to_string(i) + "_" + label;
        ++featureCounts_[featureLabel][features[i]];
        ++featureTotalCounts_[std::to_string(i)];
    }

    // Increment the total number of training examples
    ++numExamples_;
}

template <typename T>
void NaiveBayes<T>::train()
{
    // Calculate the probabilities for each label and feature
    for (const auto &labelCount : labelCounts_)
    {
        const std::string &label = labelCount.first;
        const double labelProb = static_cast<double>(labelCount.second) / static_cast<double>(numExamples_);

        for (const auto &featureTotalCount : featureTotalCounts_)
        {
            const std::string &featureLabel = featureTotalCount.first + "_" + label;
            const int totalCount = featureTotalCount.second;

            for (const auto &featureCount : featureCounts_[featureLabel])
            {
                const T &featureValue = featureCount.first;
                const int count = featureCount.second;

                const double featureProb = static_cast<double>(count) / static_cast<double>(totalCount);
                const std::string featureLabelValue = featureLabel + "_" + std::to_string(featureValue);

                featureProbs_[featureLabelValue][label] = featureProb;
            }

            const std::string totalCountKey = featureTotalCount.first + "_total";
            featureProbs_[totalCountKey][label] = static_cast<double>(totalCount) / static_cast<double>(numExamples_);
        }

        labelProbs_[label] = labelProb;
    }
}

template <typename T>
std::string NaiveBayes<T>::classify(const std::vector<T> &features)
{
    // Calculate the probabilities for each label
    std::unordered_map<std::string, double> labelProbs;
    for (const auto &labelProb : labelProbs_)
    {
        const std::string &label = labelProb.first;
        double prob = std::log(labelProb.second);

        for (size_t i = 0; i < features.size(); ++i)
        {
            const std::string featureLabelValue = std::to_string(i) + "_" + label + "_" + std::to_string(features[i]);

            const auto it = featureProbs_.find(featureLabelValue);
            if (it != featureProbs_.end())
            {
                prob += std::log(it->second[label]);
            }
        }

        labelProbs[label] = prob;
    }

    // Find the label with the highest probability
    std::string bestLabel;
    double bestProb = std::numeric_limits<double>::lowest();

    for (const auto &labelProb : labelProbs)
    {
        if (labelProb.second > bestProb)
        {
            bestLabel = labelProb.first;
            bestProb = labelProb.second;
        }
    }

    return bestLabel;
}
