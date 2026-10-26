// Copyright 2025 Intelligent Robotics Lab
//
// This file is part of the project Easy Navigation (EasyNav in short)
// licensed under the GNU General Public License v3.0.
// See <http://www.gnu.org/licenses/> for details.
//
// Easy Navigation program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

/// \file
/// \brief Implementation of the CostmapPlanner class using A* on Costmap2D.

#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>

#include "easynav_dstar_planner/DStarPlanner.hpp"

#include "nav_msgs/msg/goals.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "easynav_costmap_common/costmap_2d.hpp"
#include "easynav_costmap_common/cost_values.hpp"

namespace easynav
{

double heuristic_id(int id1, int id2, int width) {
  int x1 = id1 % width;
  int y1 = id1 / width;
  int x2 = id2 % width;
  int y2 = id2 / width;
  return std::hypot(x2 - x1, y2 - y1);
}

static std::vector<std::pair<int, int>> neighbors8 = {
  {-1, -1}, {-1, 0}, {-1, 1},
  {0, -1}, {0, 1},
  {1, -1}, {1, 0}, {1, 1}
};

static double compute_path_length(const nav_msgs::msg::Path & path)
{
  double total_length = 0.0;
  for (size_t i = 1; i < path.poses.size(); ++i) {
    const auto & p1 = path.poses[i - 1].pose.position;
    const auto & p2 = path.poses[i].pose.position;
    total_length += std::hypot(p2.x - p1.x, p2.y - p1.y);
  }
  return total_length;
}

DStarPlanner::DStarPlanner()
{
  NavState::register_printer<nav_msgs::msg::Path>(
    [](const nav_msgs::msg::Path & path) {
      std::ostringstream ret;
      ret << "Path with " << path.poses.size() << " poses and length "
          << compute_path_length(path) << " m.";
      return ret.str();
    });
}

std::expected<void, std::string> DStarPlanner::on_initialize()
{
  auto node = get_node();
  const auto & plugin_name = get_plugin_name();
  node->declare_parameter<double>(plugin_name + ".cost_factor", 2.0);
  node->declare_parameter<double>(plugin_name + ".inflation_penalty", 5.0);
  node->declare_parameter<double>(plugin_name + ".cost_axial", 1.0);
  node->declare_parameter<double>(plugin_name + ".cost_diagonal", 1.41);
  node->declare_parameter<bool>(plugin_name + ".continuous_replan", true);

  node->get_parameter(plugin_name + ".cost_factor", cost_factor_);
  node->get_parameter(plugin_name + ".inflation_penalty", inflation_penalty_);
  node->get_parameter(plugin_name + ".cost_axial", cost_axial_);
  node->get_parameter(plugin_name + ".cost_diagonal", cost_diagonal_);
  node->get_parameter(plugin_name + ".continuous_replan", continuous_replan_);

  path_pub_ = node->create_publisher<nav_msgs::msg::Path>(
    node->get_fully_qualified_name() + std::string("/") + plugin_name + "/path", 10);
  return {};
}

void DStarPlanner::update(NavState & nav_state)
{
  const auto previous_poses = current_path_.poses;
  current_path_.poses.clear();
  if (!nav_state.has("goals") || !nav_state.has("robot_pose") || !nav_state.has("map.dynamic")) {
    return;
  }

  const auto goals = nav_state.get<nav_msgs::msg::Goals>("goals");
  if (goals.goals.empty()) {
    nav_state.set("path", current_path_);
    return;
  }

  const auto & map = nav_state.get<Costmap2D>("map.dynamic");
  const auto robot_pose = nav_state.get<nav_msgs::msg::Odometry>("robot_pose");
  const auto & goal = goals.goals.front().pose;

  if (goals.header.frame_id != get_tf_prefix() + "map") {
    RCLCPP_WARN(get_node()->get_logger(), "Goals frame is not 'map': %s",
        goals.header.frame_id.c_str());
    return;
  }

  unsigned int gx, gy;
  if (!map.worldToMap(goal.position.x, goal.position.y, gx, gy)) {
    RCLCPP_WARN(get_node()->get_logger(), "Goal (%.2f, %.2f) is outside the map", goal.position.x,
        goal.position.y);
    return;
  }

  auto goals_ts = rclcpp::Time(goals.header.stamp);
  if (!continuous_replan_ &&
    goals_ts < rclcpp::Time(current_path_.header.stamp) &&
    goals.goals.front().pose == current_goal_)
  {
    return;
  }

  current_goal_ = goal;

  auto poses = d_star_path(map, robot_pose.pose.pose, goal);
  if (!poses.empty()) {
    std::unordered_set<int> previous_route;
    for (const auto & pose_stamped : previous_poses) {
      unsigned int mx, my;
      const auto & position = pose_stamped.pose.position;
      if (map.worldToMap(position.x, position.y, mx, my)) {
        previous_route.insert(static_cast<int>(my * map.getSizeInCellsX() + mx));
      }
    }

    std::unordered_set<int> current_route;
    size_t elevated_cost_cells = 0;
    unsigned int max_cell_cost = 0;
    double sum_cell_cost = 0.0;
    double route_length = 0.0;
    for (size_t i = 0; i < poses.size(); ++i) {
      const auto & position = poses[i].position;
      unsigned int mx, my;
      if (map.worldToMap(position.x, position.y, mx, my)) {
        const int cell_id = static_cast<int>(my * map.getSizeInCellsX() + mx);
        current_route.insert(cell_id);
        const unsigned int cell_cost = map.getCost(mx, my);
        max_cell_cost = std::max(max_cell_cost, cell_cost);
        sum_cell_cost += cell_cost;
        if (cell_cost >= INSCRIBED_INFLATED_OBSTACLE) {
          ++elevated_cost_cells;
        }
      }
      if (i > 0) {
        const auto & previous = poses[i - 1].position;
        route_length += std::hypot(
          position.x - previous.x, position.y - previous.y);
      }
    }

    size_t shared_cells = 0;
    for (const int cell_id : current_route) {
      if (previous_route.count(cell_id) > 0) {++shared_cells;}
    }
    const double route_overlap = current_route.empty() ? 0.0 :
      100.0 * static_cast<double>(shared_cells) / current_route.size();
    const double mean_cell_cost = current_route.empty() ? 0.0 :
      sum_cell_cost / current_route.size();
    const bool same_route = current_route == previous_route;

    RCLCPP_INFO_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 5000,
      "D* diagnóstico: ruta=%zu celdas, %.2fm; solapamiento previo=%.0f%% (%s); "
      "coste medio/máximo=%.1f/%u, celdas >=253=%zu",
      current_route.size(), route_length, route_overlap,
      same_route ? "estable" : "cambió", mean_cell_cost, max_cell_cost,
      elevated_cost_cells);

    current_path_.header.stamp = get_node()->now();
    current_path_.header.frame_id = goals.header.frame_id;
    for (const auto & pose : poses) {
      geometry_msgs::msg::PoseStamped pose_stamped;
      pose_stamped.header.frame_id = goals.header.frame_id;
      pose_stamped.header.stamp = current_path_.header.stamp;
      pose_stamped.pose = pose;
      current_path_.poses.push_back(pose_stamped);
    }
    if (path_pub_->get_subscription_count() > 0) {
      path_pub_->publish(current_path_);
    }
  }
  nav_state.set("path", current_path_);
}

std::vector<geometry_msgs::msg::Pose> DStarPlanner::d_star_path(
  const Costmap2D & map,
  const geometry_msgs::msg::Pose & start,
  const geometry_msgs::msg::Pose & goal)
{
  unsigned int sx, sy, gx, gy;
  if (!map.worldToMap(start.position.x, start.position.y, sx, sy)) {return {};}
  if (!map.worldToMap(goal.position.x, goal.position.y, gx, gy)) {return {};}

  map_width_ = map.getSizeInCellsX();
  map_height_ = map.getSizeInCellsY();

  int current_start_id = sy * map_width_ + sx;
  int current_goal_id = gy * map_width_ + gx;
  const bool goal_changed = current_goal_id != goal_id_;
  size_t changed_cost_cells = 0;
  double start_shift_cells = 0.0;

  // Objetivo nuevo?
  if (goal_changed) {
    initialize_dstar(current_start_id, current_goal_id);
    last_known_costmap_.clear();
    
    for (unsigned int i = 0; i < map.getSizeInCellsX(); ++i) {
      for (unsigned int j = 0; j < map.getSizeInCellsY(); ++j) {
        last_known_costmap_[j * map_width_ + i] = map.getCost(i, j);
      }
    }
  } else {
    int last_start_id = start_id_;
    start_id_ = current_start_id;
    start_shift_cells = heuristic_id(last_start_id, start_id_, map_width_);
    
    // Cambios?
    bool map_changed = false;
    for (unsigned int i = 0; i < map.getSizeInCellsX(); ++i) {
      for (unsigned int j = 0; j < map.getSizeInCellsY(); ++j) {
        int id = j * map_width_ + i;
        uint8_t current_cost = map.getCost(i, j);
        uint8_t last_cost;

        if (last_known_costmap_.count(id)) {
          last_cost = last_known_costmap_[id];
        } else {
          last_cost = 0;
        }

        // Cambia el coste de una celda -> obstaculo nuevo
        if (current_cost != last_cost) {
          ++changed_cost_cells;
          if (!map_changed) {
            km_ += heuristic_id(last_start_id, start_id_, map_width_);
            map_changed = true;
          }
          last_known_costmap_[id] = current_cost;
          
          update_vertex(id, map);
          for (int n_id : get_neighbors(id, map)) {
            update_vertex(n_id, map);
          }
        }
      }
    }
  }

  compute_shortest_path(map);

  if (goal_changed || changed_cost_cells > 0) {
    RCLCPP_INFO(
      get_node()->get_logger(),
      "D* entrada: meta_%s, celdas_costmap_cambiadas=%zu, movimiento_inicio=%.2f celdas, "
      "km=%.2f, cola=%zu",
      goal_changed ? "nueva" : "igual", changed_cost_cells,
      start_shift_cells, km_, open_list_.size());
  }

  std::vector<geometry_msgs::msg::Pose> path;
  int curr_id = start_id_;

  // obstaculo en el inicio o rodeado de obstaculos?
  if (!g_.count(start_id_) || g_[start_id_] == std::numeric_limits<double>::infinity()) {
    RCLCPP_WARN(get_node()->get_logger(), "¡No se encontró ruta al objetivo!");
    return {};
  }

  int loop_counter = 0;
  int max_loops = map_width_ * map_height_; // cosa que me recomendo gemini para evitar cuelgues

  while (curr_id != goal_id_ && loop_counter < max_loops) {
    double wx, wy;
    map.mapToWorld(curr_id % map_width_, curr_id / map_width_, wx, wy);
    
    geometry_msgs::msg::Pose pose;
    pose.position.x = wx;
    pose.position.y = wy;
    pose.orientation = goal.orientation;
    path.push_back(pose);

    // buscamos el vecino que esta mas cerca, tiene el menor g + coste de paso
    double min_cost = std::numeric_limits<double>::infinity();
    int best_neighbor = curr_id;

    for (int n_id : get_neighbors(curr_id, map)) {
      double edge_c = get_edge_cost(curr_id, n_id, map);
      double g_val;

      if (g_.count(n_id) > 0) {
        g_val = g_[n_id];
      } else {
        g_val = std::numeric_limits<double>::infinity();
      }

      if (edge_c + g_val < min_cost) {
        min_cost = edge_c + g_val;
        best_neighbor = n_id;
      }
    }

    // no hay vecino mejor -> atascado en un minimo local
    if (best_neighbor == curr_id) {
      RCLCPP_WARN(get_node()->get_logger(), "¡Robot atascado en un mínimo local!");
      break;
    }
    
    curr_id = best_neighbor;
    loop_counter++;
  }

  path.push_back(goal);

  return path;
}

std::vector<int> DStarPlanner::get_neighbors(int id, const Costmap2D & map) {
  std::vector<int> neighbors;
  int x = id % map_width_;
  int y = id / map_width_;

  for (auto [dx, dy] : neighbors8) {
    int nx = x + dx;
    int ny = y + dy;
    if (map.inBounds(nx, ny)) {
      neighbors.push_back(ny * map_width_ + nx);
    }
  }
  return neighbors;
}

double DStarPlanner::get_edge_cost(int id1, int id2, const Costmap2D & map) {
  int x2 = id2 % map_width_;
  int y2 = id2 / map_width_;

  uint8_t cell_cost = map.getCost(x2, y2);
  
  // hay obstacuo a donde queremos ir
  if (cell_cost >= LETHAL_OBSTACLE) {
    return std::numeric_limits<double>::infinity();
  }

  // esto lo copie del A*
  double traversal_cost = 1.0 + cost_factor_ * static_cast<double>(cell_cost) / LETHAL_OBSTACLE;
  if (cell_cost >= INSCRIBED_INFLATED_OBSTACLE) {
    double inflation_ratio = static_cast<double>(cell_cost - INSCRIBED_INFLATED_OBSTACLE) /
      (LETHAL_OBSTACLE - INSCRIBED_INFLATED_OBSTACLE);
    traversal_cost += inflation_penalty_ * inflation_ratio;
  }

  // movimiento en cruz o en diagonal?
  int x1 = id1 % map_width_;
  int y1 = id1 / map_width_;
  int dx = std::abs(x2 - x1);
  int dy = std::abs(y2 - y1);
  
  double step_cost = (dx == 0 || dy == 0) ? cost_axial_ : cost_diagonal_;

  return traversal_cost * step_cost;
}

DStarKey DStarPlanner::calculate_key(int id) {
  double g_val;
  if (g_.count(id) > 0) {
    g_val = g_[id];
  } else {
    g_val = std::numeric_limits<double>::infinity();
  }

  double rhs_val;
  if (rhs_.count(id) > 0) {
    rhs_val = rhs_[id];
  } else {
    rhs_val = std::numeric_limits<double>::infinity();
  }
  
  double min_val = std::min(g_val, rhs_val);
  
  // K1 = min(g, rhs) + heurística hacia el inicio + km (para compensar el movimiento)
  double k1 = min_val + heuristic_id(start_id_, id, map_width_) + km_;
  // K2 = min(g, rhs)
  double k2 = min_val;
  
  return {k1, k2};
}

void DStarPlanner::initialize_dstar(int start_idx, int goal_idx) {
  g_.clear();
  rhs_.clear();
  while (!open_list_.empty()) open_list_.pop();
  in_open_list_.clear();
  km_ = 0.0;

  start_id_ = start_idx;
  goal_id_ = goal_idx;

  rhs_[goal_id_] = 0.0;
  
  // empezamos a planificar desde la meta
  DStarKey start_key = calculate_key(goal_id_);
  open_list_.push({goal_id_, start_key});
  in_open_list_[goal_id_] = start_key;
}

void DStarPlanner::update_vertex(int id, const Costmap2D & map) {
  // Si no es la meta, su rhs es el de su mejor vecino mas lo que cuesta ir a él
  if (id != goal_id_) {
    double min_rhs = std::numeric_limits<double>::infinity();
    for (int n_id : get_neighbors(id, map)) {
      double n_cost = get_edge_cost(id, n_id, map);
      double g_val;

      if (g_.count(n_id)) {
        g_val = g_[n_id];
      } else {
        g_val = std::numeric_limits<double>::infinity();
      }

      min_rhs = std::min(min_rhs, n_cost + g_val);
    }
    rhs_[id] = min_rhs;
  }

  in_open_list_.erase(id);

  double g_val;
  if (g_.count(id)) {
    g_val = g_[id];
  } else {
    g_val = std::numeric_limits<double>::infinity();
  }

  double rhs_val;
  if (rhs_.count(id)) {
    rhs_val = rhs_[id];
  } else {
    rhs_val = std::numeric_limits<double>::infinity();
  }

  if (std::abs(g_val - rhs_val) > 1e-5) {
    DStarKey key = calculate_key(id);
    open_list_.push({id, key});
    in_open_list_[id] = key;
  }
}

void DStarPlanner::compute_shortest_path(const Costmap2D & map) {
  while (!open_list_.empty()) {
    DStarNode u = open_list_.top();
    
    if (in_open_list_.find(u.id) == in_open_list_.end() || in_open_list_[u.id] != u.key) {
      open_list_.pop();
      continue;
    }

    DStarKey k_new = calculate_key(u.id);
    double g_start;
    if (g_.count(start_id_)) {
      g_start = g_[start_id_];
    } else {
      g_start = std::numeric_limits<double>::infinity();
    }

    double rhs_start;
    if (rhs_.count(start_id_)) {
      rhs_start = rhs_[start_id_];
    } else {
      rhs_start = std::numeric_limits<double>::infinity();
    }

    // rompemos el bucle si me he pasado del radio del robot y esa celda no ha sufrido daños
    if (u.key >= calculate_key(start_id_) && std::abs(rhs_start - g_start) <= 1e-5) {
      break;
    }

    open_list_.pop();

    if (u.key < k_new) {
      // posible obstaculo?
      open_list_.push({u.id, k_new});
      in_open_list_[u.id] = k_new;
    } else {

      double g_val;
      if (g_.count(u.id)) {
        g_val = g_[u.id];
      } else {
        g_val = std::numeric_limits<double>::infinity();
      }

      double rhs_val;
      if (rhs_.count(u.id)) {
        rhs_val = rhs_[u.id];
      } else {
        rhs_val = std::numeric_limits<double>::infinity();
      }

      if (g_val > rhs_val) {
        // atajo (Sobre-consistente)
        g_[u.id] = rhs_val;
        in_open_list_.erase(u.id);
        for (int n_id : get_neighbors(u.id, map)) {
          update_vertex(n_id, map);
        }
      } else {
        // obstáculo (Sub-consistente)
        g_[u.id] = std::numeric_limits<double>::infinity();
        update_vertex(u.id, map);
        for (int n_id : get_neighbors(u.id, map)) {
          update_vertex(n_id, map);
        }
      }
    }
  }
}

}  // namespace easynav

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(easynav::DStarPlanner, easynav::PlannerMethodBase)