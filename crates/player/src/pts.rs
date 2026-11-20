#[derive(Default)]
pub(crate) struct PtsQueue(Vec<u64>);

impl PtsQueue {
    pub(crate) fn push(&mut self, pts: u64) {
        let at = self.0.partition_point(|&queued| queued <= pts);
        self.0.insert(at, pts);
    }

    pub(crate) fn pop_earliest(&mut self) -> Option<u64> {
        (!self.0.is_empty()).then(|| self.0.remove(0))
    }

    pub(crate) fn remove(&mut self, pts: u64) {
        if let Ok(at) = self.0.binary_search(&pts) {
            self.0.remove(at);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earliest_pts_comes_out_first_whatever_the_decode_order() {
        let mut queue = PtsQueue::default();
        for pts in [0, 300, 100, 200] {
            queue.push(pts);
        }
        let order: Vec<_> = std::iter::from_fn(|| queue.pop_earliest()).collect();
        assert_eq!(order, [0, 100, 200, 300]);
    }

    #[test]
    fn equal_pts_are_kept_once_each() {
        let mut queue = PtsQueue::default();
        queue.push(5);
        queue.push(5);
        assert_eq!(queue.pop_earliest(), Some(5));
        assert_eq!(queue.pop_earliest(), Some(5));
        assert_eq!(queue.pop_earliest(), None);
    }

    #[test]
    fn a_failed_packet_gives_its_pts_back() {
        let mut queue = PtsQueue::default();
        queue.push(10);
        queue.push(20);
        queue.remove(10);
        queue.remove(99);
        assert_eq!(queue.pop_earliest(), Some(20));
    }
}
