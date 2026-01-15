#include "LRUCache.hpp"


LRUCache::LRUCache(size_t max_total_experts, size_t per_expert_size, EvictCallback cb)
    : m_max_total_experts(max_total_experts),
      m_per_expert_size(per_expert_size),
      m_total_experts(0),
      m_on_evict(std::move(cb)) {
        m_filled_list.resize(max_total_experts, false);
      }

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end())
        return;
    m_list.splice(m_list.end(), m_list, it);
}

void LRUCache::evict_one() {
    if (m_list.empty()) return;

    auto& oldest = m_list.front();

    m_filled_list[oldest.lru_expert_no] = false;
    Key key{oldest.layer, oldest.expert};
    m_map.erase(key);
    m_list.pop_front();
    --m_total_experts;
}

std::pair<size_t, bool> LRUCache::get_lru_item(size_t layer, size_t expert) {
   Key key{layer, expert};
   auto it = m_map.find(key);
   if (it == m_map.end()) {
       if (m_total_experts > m_max_total_experts) {
           evict_one();
       }
       auto lru_expert_no = m_list.size();
       m_list.push_back(Node{layer, expert, lru_expert_no});
       auto new_it = std::prev(m_list.end());
       m_map[key] = new_it;
       ++m_total_experts;
   } else {
       move_to_end(it->second);
   }
   return { it->second->lru_expert_no, m_filled_list[it->second->lru_expert_no] };
}


// void* LRUCache::get_expert_addr(size_t layer, size_t expert) {
//     Key key{layer, expert};
//     auto it = m_map.find(key);
//     if (it == m_map.end()) return nullptr;
//     move_to_end(it->second);
//     return it->second->addr;
// }

// void* LRUCache::get_expert_params(size_t layer, size_t expert) {
//     Key key{layer, expert};
//     auto it = m_map.find(key);
//     if (it == m_map.end()) return nullptr;
//     move_to_end(it->second);
//     return it->second->params;
// }