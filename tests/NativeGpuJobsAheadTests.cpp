#include "graphics/native_gpu/nativeJobsAhead.h"

#include <cstdio>
#include <cstdlib>
#include <deque>
#include <vector>

using Libs::Graphics::NativeGpu::TakeJobsAhead;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NativeGpuJobsAheadTests: failed: %s\n", message);
		std::abort();
	}
}

struct Job {
	char kind; // 'a' compute, 'd' graphics, 'k' Kyty's own
	int  id;
};

bool Takes(const Job& j) {
	return j.kind == 'a';
}
bool Stops(const Job& j) {
	return j.kind == 'k';
}

std::vector<int> Ids(const std::vector<Job>& jobs) {
	std::vector<int> ids;
	for (const Job& j: jobs) {
		ids.push_back(j.id);
	}
	return ids;
}

std::vector<int> Ids(const std::deque<Job>& jobs) {
	return Ids(std::vector<Job>(jobs.begin(), jobs.end()));
}

void TestTakesInOrderPastGraphics() {
	std::deque<Job>  from {{'a', 1}, {'d', 2}, {'a', 3}, {'a', 4}, {'d', 5}};
	std::vector<Job> to;
	Check(TakeJobsAhead(from, &to, Takes, Stops), "nothing stops it");
	Check(Ids(to) == std::vector<int>({1, 3, 4}), "compute buffers keep their order and pass graphics ones");
	Check(Ids(from) == std::vector<int>({2, 5}), "graphics buffers stay, in order");
}

void TestStopsAtKytyOwn() {
	std::deque<Job>  from {{'a', 1}, {'k', 2}, {'a', 3}};
	std::vector<Job> to;
	Check(!TakeJobsAhead(from, &to, Takes, Stops), "reports the stop, so the next deque is left alone");
	Check(Ids(to) == std::vector<int>({1}), "nothing passes Kyty's own submission");
	Check(Ids(from) == std::vector<int>({2, 3}), "it and what follows stay");
}

void TestAppends() {
	std::deque<Job>  first {{'a', 1}};
	std::deque<Job>  second {{'a', 2}};
	std::vector<Job> to;
	Check(TakeJobsAhead(first, &to, Takes, Stops) && TakeJobsAhead(second, &to, Takes, Stops), "two deques");
	Check(Ids(to) == std::vector<int>({1, 2}), "the second deque's come after the first's");
	Check(first.empty() && second.empty(), "both emptied");
}

void TestEmpty() {
	std::deque<Job>  from;
	std::vector<Job> to;
	Check(TakeJobsAhead(from, &to, Takes, Stops) && to.empty(), "an empty deque");
}

} // namespace

int main() {
	TestTakesInOrderPastGraphics();
	TestStopsAtKytyOwn();
	TestAppends();
	TestEmpty();
	std::fprintf(stderr, "NativeGpuJobsAheadTests: all passed\n");
	return 0;
}
