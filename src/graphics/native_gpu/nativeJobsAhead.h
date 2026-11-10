#pragma once

#include <deque>
#include <utility>
#include <vector>

namespace Libs::Graphics::NativeGpu {

// Moves the jobs `takes` accepts from `from` to the end of `to`, keeping their order, and stops at the first job
// `stops` names: nothing behind it may pass it. False when it stopped, so a deque of later jobs is left alone.
template <class Job, class Takes, class Stops>
bool TakeJobsAhead(std::deque<Job>& from, std::vector<Job>* to, Takes takes, Stops stops) {
	for (auto it = from.begin(); it != from.end();) {
		if (stops(*it)) {
			return false;
		}
		if (takes(*it)) {
			to->push_back(std::move(*it));
			it = from.erase(it);
		} else {
			++it;
		}
	}
	return true;
}

} // namespace Libs::Graphics::NativeGpu
