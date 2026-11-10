#pragma once

#include <cstdint>

namespace Libs::Graphics::NativeGpu {

// The order between the compute buffers a thread of their own translates and Kyty's own submissions (null buffers),
// which the graphics worker takes in turn. A compute buffer does not start before every null enqueued before it has
// been passed; a null is not passed before every compute buffer enqueued before it has been handed to the worker.
// Not synchronized: the caller holds the lock around enqueueing and the two threads' decisions.
class AcbNullGate {
public:
	// The stamp of a compute buffer being enqueued: the nulls enqueued before it.
	uint64_t EnqueueCompute() {
		m_computes_enqueued++;
		return m_nulls_enqueued;
	}
	// The stamp of a null being enqueued: the compute buffers enqueued before it.
	uint64_t EnqueueNull() {
		m_nulls_enqueued++;
		return m_computes_enqueued;
	}
	[[nodiscard]] bool ComputeMayStart(uint64_t compute_stamp) const { return m_nulls_passed >= compute_stamp; }
	[[nodiscard]] bool NullMayGo(uint64_t null_stamp) const { return m_computes_handed >= null_stamp; }
	// A compute buffer is the worker's now, refused or not.
	void ComputeHanded() { m_computes_handed++; }
	void NullPassed() { m_nulls_passed++; }

private:
	uint64_t m_computes_enqueued = 0;
	uint64_t m_computes_handed   = 0;
	uint64_t m_nulls_enqueued    = 0;
	uint64_t m_nulls_passed      = 0;
};

} // namespace Libs::Graphics::NativeGpu
