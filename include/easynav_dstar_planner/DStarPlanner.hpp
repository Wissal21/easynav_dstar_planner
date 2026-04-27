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
/// \brief Declaration of the CostmapPlanner class implementing A* path planning using Costmap2D.

#ifndef EASYNAV_DSTAR_PLANNER__DSTARPLANNER_HPP_
#define EASYNAV_DSTAR_PLANNER__DSTARPLANNER_HPP_

#include <memory>
#include <vector>

#include "nav_msgs/msg/path.hpp"

#include "easynav_core/PlannerMethodBase.hpp"
#include "easynav_costmap_common/costmap_2d.hpp"
#include "easynav_common/types/NavState.hpp"

namespace easynav
{

// Clave de prioridad para D*
struct DStarKey {
  double k1;
  double k2;

  bool operator<(const DStarKey& other) const {
    if (k1 == other.k1) return k2 < other.k2;
    return k1 < other.k1;
  }
  bool operator>(const DStarKey& other) const {
    if (k1 == other.k1) return k2 > other.k2;
    return k1 > other.k1;
  }
  bool operator<=(const DStarKey& other) const {
    return *this < other || (*this == other);
  }
  bool operator>=(const DStarKey& other) const {
    return *this > other || (*this == other);
  }
  bool operator==(const DStarKey& other) const {
    return k1 == other.k1 && k2 == other.k2;
  }
};

// Estructura para los nodos en la cola
struct DStarNode {
  int id;
  DStarKey key;

  bool operator>(const DStarNode& other) const {
    return key > other.key;
  }
};

/// \brief A planner implementing the A* algorithm on a Costmap2D grid.
///
/// This class generates a collision-free path using A* search over a 2D costmap.
/// It supports cost-based penalties and anisotropic movement costs.
class DStarPlanner : public PlannerMethodBase
{
public:
  /**
   * @brief Default constructor.
   *
   * Initializes internal parameters and configuration values.
   */
  explicit DStarPlanner();

  /**
   * @brief Initializes the planner.
   *
   * Loads planner parameters, sets up ROS publishers,
   * and prepares the costmap-based planning environment.
   *
   * @return std::expected<void, std::string> A success indicator or error message.
   */
  virtual std::expected<void, std::string> on_initialize() override;

  /**
   * @brief Executes a planning cycle using the current navigation state.
   *
   * Computes a path from the robot's current pose to the goal using A*.
   *
   * @param nav_state Current shared navigation state (input/output).
   */
  void update(NavState & nav_state) override;

protected:
  double cost_factor_;        ///< Scaling factor applied to cell cost values.
  double inflation_penalty_; ///< Extra cost penalty for paths near inflated obstacles.
  double cost_axial_;        ///< Cost multiplier for axial (horizontal/vertical) moves.
  double cost_diagonal_;     ///< Cost multiplier for diagonal moves.
  bool continuous_replan_ {true};    ///< Wheter replan path at freq time
  nav_msgs::msg::Path current_path_;  ///< Most recently computed path.
  geometry_msgs::msg::Pose current_goal_;  ///< Current goal.

  /// Publisher for the computed navigation path (for visualization or monitoring).
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  /**
   * @brief Internal A* path planning routine.
   *
   * Computes a path on the given costmap from the start pose to the goal pose.
   *
   * Movement cost is influenced by:
   * - The cost of each cell (retrieved from the costmap).
   * - Additional inflation penalties near obstacles.
   * - Anisotropic weights for axial vs diagonal movement.
   *
   * @param map The costmap to plan over.
   * @param start The robot's starting pose in world coordinates.
   * @param goal The goal pose in world coordinates.
   * @return A vector of poses representing the planned path.
   */
  std::vector<geometry_msgs::msg::Pose> d_star_path(
    const Costmap2D & map,
    const geometry_msgs::msg::Pose & start,
    const geometry_msgs::msg::Pose & goal);

  // Inicializa las estructuras si es la primera vez o si la meta cambia
  void initialize_dstar(int start_idx, int goal_idx);

  // Calcula la clave (k1, k2) para una celda
  DStarKey calculate_key(int id);

  // Actualiza el valor rhs de una celda y lo mete/saca de la cola según corresponda
  void update_vertex(int id, const Costmap2D & map);

  // El bucle principal que procesa la cola hasta que el inicio es consistente
  void compute_shortest_path(const Costmap2D & map);

  // Devuelve los vecinos válidos (ID) de una celda dada
  std::vector<int> get_neighbors(int id, const Costmap2D & map);

  // Devuelve el coste de moverse entre dos celdas vecinas
  double get_edge_cost(int id1, int id2, const Costmap2D & map);

  // Mapas para almacenar los valores g y rhs de cada celda
  std::unordered_map<int, double> g_;
  std::unordered_map<int, double> rhs_;

  // La cola de prioridad
  std::priority_queue<DStarNode, std::vector<DStarNode>, std::greater<DStarNode>> open_list_;

  // Lista auxiliar para usar la cola de prioridad correctamente
  std::unordered_map<int, DStarKey> in_open_list_;

  // Punteros para recordar el inicio y la meta anteriores
  int start_id_{-1};
  int goal_id_{-1};

  // Variable km de D* (acumula el coste de los movimientos del robot para mantener las claves consistentes)
  double km_{0.0};

  // Guardamos una copia de los costes anteriores para detectar qué obstáculos han cambiado
  std::unordered_map<int, uint8_t> last_known_costmap_;

  // Información del mapa
  int map_width_{0};
  int map_height_{0};

};

}  // namespace easynav

#endif  // EASYNAV_DSTAR_PLANNER__DSTARPLANNER_HPP_