// TODO: Use pragma once
#ifndef BALL_ANALYSIS_CLASSIFICATION_H
#define BALL_ANALYSIS_CLASSIFICATION_H

#include <stddef.h>
#include <stdint.h>

namespace BallClassifier {

/**
 * @brief Number of sensor features expected by the classifier.
 *
 * Every prediction function in this module expects a 10-dimensional input
 * vector that matches the AS7341 feature layout used during training.
 */
constexpr size_t kFeatureCount = 10;

/**
 * @brief Number of known ball colors embedded in the classifier.
 */
constexpr size_t kClassCount = 7;

/**
 * @brief Euclidean distance threshold used by the unknown-aware classifier.
 *
 * Inputs farther than this threshold from their closest normalized class
 * centroid are rejected as `unknown`.
 */
constexpr float kUnknownThreshold = 0.06472885f;

/**
 * @brief Predict the closest known ball color from a floating-point feature vector.
 *
 * The input vector is normalized internally before classification, so raw sensor
 * magnitudes may be passed directly as long as the feature order matches the
 * model training data.
 *
 * @param input A 10-dimensional feature vector.
 * @return One of `orange`, `purple`, `blue`, `green`, `yellow`, `pink`, `red`,
 *         or `unknown` if the input cannot be normalized.
 */
const char* classifyBallColor(const float input[kFeatureCount]);

/**
 * @brief Predict the closest known ball color from a 16-bit integer feature vector.
 *
 * This overload is convenient for Arduino sketches that store AS7341 readings in
 * integer arrays.
 *
 * @param input A 10-dimensional feature vector.
 * @return One of `orange`, `purple`, `blue`, `green`, `yellow`, `pink`, `red`,
 *         or `unknown` if the input cannot be normalized.
 */
const char* classifyBallColor(const uint16_t input[kFeatureCount]);

/**
 * @brief Predict the ball color and reject outliers as `unknown`.
 *
 * The input is normalized internally and compared against the stored class
 * centroids. If the closest centroid is farther than `kUnknownThreshold`, the
 * function returns `unknown` instead of forcing a known label.
 *
 * @param input A 10-dimensional feature vector.
 * @return One of `orange`, `purple`, `blue`, `green`, `yellow`, `pink`, `red`,
 *         or `unknown`.
 */
const char* classifyBallColorOrUnknown(const float input[kFeatureCount]);

/**
 * @brief Integer-input overload of the unknown-aware classifier.
 *
 * @param input A 10-dimensional feature vector.
 * @return One of `orange`, `purple`, `blue`, `green`, `yellow`, `pink`, `red`,
 *         or `unknown`.
 */
const char* classifyBallColorOrUnknown(const uint16_t input[kFeatureCount]);

}  // namespace BallClassifier

#endif  // BALL_ANALYSIS_CLASSIFICATION_H