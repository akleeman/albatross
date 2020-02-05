/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_INDEXING_APPLY_HPP_
#define ALBATROSS_INDEXING_APPLY_HPP_

namespace albatross {

struct ApplyRange {
  std::size_t start;
  std::size_t end;
  ApplyRange(const std::size_t end_) : start(0), end(end_){};
  ApplyRange(const std::size_t start_, const std::size_t end_)
      : start(start_), end(end_){};
};

// Range

template <typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, std::size_t>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, std::size_t>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
void apply(const ApplyRange &range, const ApplyFunction &f) {
  for (std::size_t i = range.start; i < range.end; ++i) {
    f(i);
  }
}

// Vector
template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
void apply(const std::vector<ValueType> &xs, const ApplyFunction &f) {
  std::for_each(xs.begin(), xs.end(), f);
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
auto apply(const std::vector<ValueType> &xs, const ApplyFunction &f) {
  std::vector<ApplyType> output(xs.size());
  std::transform(xs.begin(), xs.end(), output.begin(), f);
  return output;
}

// Map

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ApplyFunction,
    typename ApplyType = typename details::key_value_apply_result<
        ApplyFunction, KeyType, ValueType>::type,
    typename std::enable_if<details::is_valid_key_value_apply_function<
                                ApplyFunction, KeyType, ValueType>::value &&
                                std::is_same<void, ApplyType>::value,
                            int>::type = 0>
void apply(const Map<KeyType, ValueType> &map, const ApplyFunction &f) {
  for (const auto &pair : map) {
    f(pair.first, pair.second);
  }
}

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ApplyFunction,
    typename ApplyType = typename details::key_value_apply_result<
        ApplyFunction, KeyType, ValueType>::type,
    typename std::enable_if<details::is_valid_key_value_apply_function<
                                ApplyFunction, KeyType, ValueType>::value &&
                                !std::is_same<void, ApplyType>::value,
                            int>::type = 0>
auto apply(const Map<KeyType, ValueType> &map, const ApplyFunction &f) {
  Grouped<KeyType, ApplyType> output;
  for (const auto &pair : map) {
    output.emplace(pair.first, f(pair.first, pair.second));
  }
  return output;
}

template <template <typename...> class Map, typename KeyType,
          typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
auto apply(const Map<KeyType, ValueType> &map, const ApplyFunction &f) {
  Grouped<KeyType, ApplyType> output;
  for (const auto &pair : map) {
    output.emplace(pair.first, f(pair.second));
  }
  return output;
}

template <template <typename...> class Map, typename KeyType,
          typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
auto apply(const Map<KeyType, ValueType> &map, const ApplyFunction &f) {
  for (const auto &pair : map) {
    f(pair.second);
  }
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_APPLY_HPP_ */
