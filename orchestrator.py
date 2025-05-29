# Gerekli Kütüphaneler
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import heapq # A* için öncelik kuyruğu
import matplotlib.animation as animation # Animasyon için
import xml.etree.ElementTree as ET # SUMO XML için
from xml.dom import minidom      # SUMO XML pretty print için

# sumolib'i import etmeye çalışalım
try:
    import sumolib
    SUMOLIB_AVAILABLE = True
    print("sumolib kütüphanesi başarıyla yüklendi.")
except ImportError:
    SUMOLIB_AVAILABLE = False
    print("UYARI: sumolib kütüphanesi bulunamadı. SUMO kenar ID dönüşümü daha az güvenilir olabilir veya çalışmayabilir.")

print("Faz 1, 2, 3 (A*), 4 (Animasyon) & SUMO Rota Çevirme (sumolib ile - Strateji B+C)")

# --- CA ve Simülasyon Ayarları ---
DELTA_T = 1.0
ACCELERATION = 1.0
DECELERATION = 3.5
P_SLOWDOWN = 0.05
DEFAULT_SPEED_LIMIT_KMPH = 20
METER_PER_KM = 1000
SECONDS_PER_HOUR = 3600
MAX_SIM_TIME_STEPS = 250 
MAX_ASTAR_TIME_STEPS_FACTOR = 5 

# --- Global Değişkenler ---
reservations = {} 
fig_anim, ax_anim = None, None 
agent_artists = [] 
sim_agents_list_anim = [] 
current_anim_time = 0.0 
G_osmnx_planning_graph = None 
route_colors_global = ['r', 'b', 'darkorange', 'fuchsia', 'c', 'm', 'lime', 'gold']
sumo_net_obj_global = None 
ani = None 
SUMO_EDGES_BY_NODES_CACHE = {} # Strateji B için önbellek
ORIG_ID_TO_SUMO_EDGES_MAP = {} # Strateji C için harita


# --- Yardımcı Fonksiyonlar ---
def get_location_id(node=None, edge=None):
    if node is not None: return int(node)
    if edge is not None: return tuple(sorted(edge))
    return None
def reserve_location(agent_id, time_step, location_id):
    ts = int(round(time_step / DELTA_T));
    if ts < 0 : ts = 0
    if ts not in reservations: reservations[ts] = {}
    existing_agent = reservations[ts].get(location_id, None)
    if existing_agent is not None and existing_agent != agent_id: return False 
    reservations[ts][location_id] = agent_id
    return True
def is_location_reserved(time_step, location_id, exclude_agent_id=None):
    ts = int(round(time_step / DELTA_T));
    if ts < 0 : ts = 0
    if ts not in reservations or location_id not in reservations[ts]: return False
    if exclude_agent_id is not None and reservations[ts][location_id] == exclude_agent_id: return False
    return True
def get_edge_speed_limit_mps(G, u, v):
    try:
        edge_data = G.get_edge_data(u, v, 0) 
        if edge_data is None: return DEFAULT_SPEED_LIMIT_KMPH * METER_PER_KM / SECONDS_PER_HOUR
        speed_str = edge_data.get('maxspeed', str(DEFAULT_SPEED_LIMIT_KMPH))
        if isinstance(speed_str, list): speed_str = speed_str[0]
        try: speed_kmph = int(str(speed_str).split()[0])
        except ValueError: speed_kmph = DEFAULT_SPEED_LIMIT_KMPH
        return speed_kmph * METER_PER_KM / SECONDS_PER_HOUR
    except Exception: return DEFAULT_SPEED_LIMIT_KMPH * METER_PER_KM / SECONDS_PER_HOUR

# --- Agent Sınıfı ---
class Agent: # (Değişiklik yok)
    def __init__(self, agent_id, start_node, goal_node, G_graph):
        self.id = agent_id; self.G = G_graph; self.start_node = start_node; self.goal_node = goal_node
        self.current_node = start_node; self.on_edge = None; self.edge_progress = 0.0
        self.current_location_id = get_location_id(node=self.current_node)
        self.path_nodes = []; self.path_index = 0; self.speed = 0.0; self.status = "NOT_STARTED"
    def set_path(self, path_node_list):
        if not path_node_list or path_node_list[0] != self.start_node:
            self.path_nodes = []; self.status = "ERROR_INVALID_PATH"; return
        self.path_nodes = path_node_list; self.path_index = 0
        if len(self.path_nodes) > 1:
            self.status = "MOVING"; self.speed = 0.0; self.on_edge = None
            self.edge_progress = 0.0; self.current_node = self.start_node
            self.current_location_id = get_location_id(node=self.current_node)
        elif len(self.path_nodes) == 1 and self.start_node == self.goal_node: self.status = "ARRIVED"
        else: self.status = "ERROR_SHORT_PATH"
    def __repr__(self):
        loc_str = f"N{self.current_node}" if self.on_edge is None else f"E{self.on_edge}({self.edge_progress:.0f}m)"
        path_str = f"{self.path_index}/{len(self.path_nodes)-1}" if self.path_nodes and len(self.path_nodes)>1 else ("0/0" if self.path_nodes else "N/A")
        return f"A{self.id}[{loc_str} S:{self.speed:.1f} D:{self.status} P:{path_str} H:{self.goal_node}]"
    def update_state(self, current_time): # (Değişiklik yok)
        if self.status != "MOVING": reserve_location(self.id, current_time + DELTA_T, self.current_location_id); return
        target_node = None; current_edge_data = None
        v_max_road = DEFAULT_SPEED_LIMIT_KMPH*METER_PER_KM/SECONDS_PER_HOUR; edge_len = 1.0
        if self.on_edge:
            u, v = self.on_edge
            try:
                current_edge_data = self.G.get_edge_data(u,v,0);
                if current_edge_data is None: raise KeyError(f"A{self.id}: {u}->{v} on_edge'de veri yok")
                v_max_road=get_edge_speed_limit_mps(self.G,u,v); edge_len=current_edge_data.get('length',1.0); target_node=v
            except Exception as e: self.status="ERROR"; print(f"A{self.id} update_state on_edge hata: {e}"); reserve_location(self.id,current_time+DELTA_T,self.current_location_id); return
        elif self.path_index < len(self.path_nodes)-1:
            u=self.current_node; v=self.path_nodes[self.path_index+1]
            try:
                current_edge_data=self.G.get_edge_data(u,v,0) 
                if current_edge_data is None: raise KeyError(f"A{self.id}: {u}->{v} yeni kenar verisi yok")
                self.on_edge=(u,v); self.edge_progress=0.0; v_max_road=get_edge_speed_limit_mps(self.G,u,v)
                edge_len=current_edge_data.get('length',1.0); target_node=v
                self.current_location_id=get_location_id(edge=self.on_edge)
            except Exception as e: self.status="ERROR"; print(f"A{self.id} update_state new_edge hata: {e}"); reserve_location(self.id,current_time+DELTA_T,self.current_location_id); return
        else: self.status="ARRIVED"; self.speed=0.0; reserve_location(self.id,current_time+DELTA_T,self.current_location_id); return
        target_speed=min(self.speed+ACCELERATION*DELTA_T, v_max_road)
        lookahead_time=current_time+DELTA_T; next_major_location_id=None
        remaining_dist_on_edge=edge_len-self.edge_progress if edge_len > self.edge_progress else 0.0
        if self.on_edge and target_speed*DELTA_T >= remaining_dist_on_edge: next_major_location_id=get_location_id(node=target_node)
        elif self.on_edge: next_major_location_id=get_location_id(edge=self.on_edge)
        else: next_major_location_id=self.current_location_id
        safe_speed=target_speed
        if next_major_location_id and is_location_reserved(lookahead_time,next_major_location_id,exclude_agent_id=self.id): safe_speed=0.0
        target_speed=max(0.0,safe_speed)
        if random.random()<P_SLOWDOWN and target_speed>0: target_speed=max(0.0,target_speed-DECELERATION*DELTA_T*random.random())
        self.speed=target_speed; distance_to_move=self.speed*DELTA_T
        if not self.on_edge: reserve_location(self.id,current_time+DELTA_T,self.current_location_id); return
        u,v=self.on_edge; next_reservation_location=self.current_location_id
        if distance_to_move>=remaining_dist_on_edge:
            time_to_reach_node=remaining_dist_on_edge/self.speed if self.speed>0 else float('inf')
            if time_to_reach_node==np.inf and remaining_dist_on_edge > 0 : self.edge_progress+=0; reserve_location(self.id,current_time+DELTA_T,self.current_location_id); return
            self.current_node=v; self.on_edge=None; self.edge_progress=0.0; self.path_index+=1
            self.current_location_id=get_location_id(node=self.current_node); next_reservation_location=self.current_location_id
            if self.current_node==self.goal_node and self.path_index>=len(self.path_nodes)-1: self.status="ARRIVED"; self.speed=0.0
        else: self.edge_progress+=distance_to_move; next_reservation_location=self.current_location_id
        reserve_location(self.id,current_time+DELTA_T,next_reservation_location)


# --- Zaman Uyumlu A* Fonksiyonları ---
def calculate_heuristic(node1_id, node2_id, G): # (Değişiklik yok)
    if node1_id not in G.nodes or node2_id not in G.nodes: return float('inf')
    node1_data = G.nodes[node1_id]; node2_data = G.nodes[node2_id]
    try: distance = np.sqrt((node1_data['x'] - node2_data['x'])**2 + (node1_data['y'] - node2_data['y'])**2)
    except KeyError: return 0 
    avg_speed_mps = (DEFAULT_SPEED_LIMIT_KMPH*METER_PER_KM/SECONDS_PER_HOUR)*0.8 
    if avg_speed_mps <= 0: avg_speed_mps = 10.0
    return distance / avg_speed_mps
def find_timed_path_A_star(agent_id, start_node, goal_node, graph, constraints, start_time=0.0): # (Değişiklik yok)
    max_ts_astar = MAX_SIM_TIME_STEPS * MAX_ASTAR_TIME_STEPS_FACTOR
    start_ts_idx = int(round(start_time/DELTA_T))
    if (get_location_id(node=start_node), start_ts_idx) in constraints: return None
    open_set = []; h_initial = calculate_heuristic(start_node, goal_node, graph)
    if h_initial == float('inf'): return None
    heapq.heappush(open_set, (start_time + h_initial, start_time, start_ts_idx, start_node, [(start_node,start_time)]))
    closed_set = {}
    while open_set:
        f,g,ts,curr_n,path = heapq.heappop(open_set)
        if curr_n == goal_node: return path
        if ts >= max_ts_astar: continue
        state_t = (curr_n,ts);
        if state_t in closed_set and closed_set[state_t] <= g: continue
        closed_set[state_t] = g
        possible_next_states = []
        next_n_w, next_ts_w = curr_n, ts+1
        if not (get_location_id(node=next_n_w), next_ts_w) in constraints and next_ts_w < max_ts_astar:
            new_g_w = g+DELTA_T
            if closed_set.get((next_n_w,next_ts_w), float('inf')) > new_g_w:
                h_w=calculate_heuristic(next_n_w,goal_node,graph)
                if h_w != float('inf'): possible_next_states.append((new_g_w+h_w,new_g_w,next_ts_w,next_n_w, path+[(next_n_w,next_ts_w*DELTA_T)]))
        for neighbor_n in graph.successors(curr_n) if graph.is_directed() else graph.neighbors(curr_n): 
            try:
                edge_d = graph.get_edge_data(curr_n,neighbor_n,0) 
                if not edge_d: continue 
                edge_l=edge_d.get('length',1.0); edge_l=max(1.0,edge_l)
                speed_l_mps=get_edge_speed_limit_mps(graph,curr_n,neighbor_n)
                time_on_e=edge_l/speed_l_mps if speed_l_mps>0 else float('inf')
                move_s=max(1,int(np.ceil(time_on_e/DELTA_T))); move_t_sec=move_s*DELTA_T
                next_n_m, next_ts_m = neighbor_n, ts+move_s
                if not (get_location_id(node=next_n_m), next_ts_m) in constraints and next_ts_m < max_ts_astar:
                    new_g_m = g+move_t_sec
                    if closed_set.get((next_n_m,next_ts_m), float('inf')) > new_g_m:
                        h_m=calculate_heuristic(next_n_m,goal_node,graph)
                        if h_m != float('inf'): possible_next_states.append((new_g_m+h_m,new_g_m,next_ts_m,next_n_m, path+[(next_n_m,next_ts_m*DELTA_T)]))
            except: continue
        for state_data in possible_next_states: heapq.heappush(open_set, state_data)
    return None

# --- CTNode Sınıfı ve Çakışma Tespiti ---
class CTNode: # (Değişiklik yok)
    def __init__(self, constraints, solution):
        self.constraints = constraints; self.solution = solution; self.cost = self.calculate_cost()
    def calculate_cost(self):
        tc=0;
        if not self.solution: return float('inf')
        for p in self.solution.values():
            if p and p[-1][1] is not None : tc+=p[-1][1] 
            else: return float('inf')
        return tc
    def __lt__(self, other):
        if self.cost != other.cost: return self.cost < other.cost
        return sum(len(c) for c in self.constraints.values()) < sum(len(c) for c in other.constraints.values())
    def __repr__(self): return f"CTN(C:{self.cost:.0f} #Cons:{sum(len(c) for c in self.constraints.values())})"
def find_first_conflict(solution): # (Değişiklik yok)
    if not solution or len(solution)<2: return None
    max_t=0;
    for p in solution.values():
        if p: max_t=max(max_t,p[-1][1])
    max_ts=int(round(max_t/DELTA_T)); occ_map={}
    for aid,p in solution.items():
        if not p: continue
        for i in range(len(p)):
            nid,t_s=p[i]; ts_idx=int(round(t_s/DELTA_T)); loc_id=get_location_id(node=nid)
            if ts_idx in occ_map and loc_id in occ_map[ts_idx] and occ_map[ts_idx][loc_id]!=aid:
                return {'agent1':aid,'agent2':occ_map[ts_idx][loc_id],'location':loc_id,'time_step':ts_idx}
            if ts_idx not in occ_map: occ_map[ts_idx]={}
            occ_map[ts_idx][loc_id]=aid;
            if i+1<len(p):
                next_nid,next_t_s=p[i+1]; next_ts_idx=int(round(next_t_s/DELTA_T))
                for w_ts_idx in range(ts_idx+1,next_ts_idx): 
                    if w_ts_idx in occ_map and loc_id in occ_map[w_ts_idx] and occ_map[w_ts_idx][loc_id]!=aid:
                        return {'agent1':aid,'agent2':occ_map[w_ts_idx][loc_id],'location':loc_id,'time_step':w_ts_idx}
                    if w_ts_idx not in occ_map: occ_map[w_ts_idx]={}
                    occ_map[w_ts_idx][loc_id]=aid;
    return None

# --- Conflict-Based Search (CBS) Ana Fonksiyonu ---
def run_cbs(initial_agents_list, graph, max_cbs_iterations=50): # (Değişiklik yok)
    print("\n--- CBS Başlatılıyor ---"); cbs_open=[]; iter_c=0
    init_cons={a.id:set() for a in initial_agents_list}; init_sol={}; all_init_ok=True
    for a in initial_agents_list:
        p_tuples=find_timed_path_A_star(a.id,a.start_node,a.goal_node,graph,init_cons[a.id])
        if p_tuples: init_sol[a.id]=p_tuples
        else: all_init_ok=False; print(f"CBS KÖK HATA: Agent {a.id} için A* yol bulamadı."); break
    if not all_init_ok: return None
    root_ctn=CTNode(init_cons,init_sol)
    if root_ctn.cost==float('inf'): print("CBS HATA: Kök maliyeti sonsuz."); return None
    heapq.heappush(cbs_open,root_ctn); 
    while cbs_open and iter_c<max_cbs_iterations:
        iter_c+=1; curr_ctn=heapq.heappop(cbs_open)
        conflict=find_first_conflict(curr_ctn.solution)
        if conflict is None: print(f"CBS BAŞARILI! {iter_c} iterasyonda. Maliyet: {curr_ctn.cost:.0f}"); return curr_ctn.solution
        for aid_to_constrain in [conflict['agent1'],conflict['agent2']]:
            new_c=(conflict['location'],conflict['time_step'])
            child_cons={aid:c.copy() for aid,c in curr_ctn.constraints.items()}
            if aid_to_constrain not in child_cons: child_cons[aid_to_constrain]=set()
            if new_c in child_cons[aid_to_constrain]: continue
            child_cons[aid_to_constrain].add(new_c)
            cons_agent_obj=next((ag for ag in initial_agents_list if ag.id==aid_to_constrain),None)
            if not cons_agent_obj: continue
            new_p_constrained=find_timed_path_A_star(aid_to_constrain,cons_agent_obj.start_node,cons_agent_obj.goal_node,graph,child_cons[aid_to_constrain])
            if new_p_constrained:
                child_sol=curr_ctn.solution.copy(); child_sol[aid_to_constrain]=new_p_constrained
                child_ctn_new=CTNode(child_cons,child_sol) 
                if child_ctn_new.cost!=float('inf'): heapq.heappush(cbs_open,child_ctn_new)
    if iter_c>=max_cbs_iterations: print(f"CBS BAŞARISIZ: Maks iterasyon ({max_cbs_iterations}) aşıldı.")
    else: print("CBS BAŞARISIZ: OPEN boşaldı, çakışmasız çözüm bulunamadı.")
    return None

# --- Animasyon Fonksiyonları ---
def init_animation_func(): # (Değişiklik yok)
    global agent_artists, sim_agents_list_anim, current_anim_time, G_osmnx_planning_graph, route_colors_global
    reservations.clear(); current_anim_time = 0.0
    for i, agent_sim in enumerate(sim_agents_list_anim):
        agent_sim.current_node = agent_sim.start_node; agent_sim.on_edge = None
        agent_sim.edge_progress = 0.0; agent_sim.speed = 0.0; agent_sim.path_index = 0
        agent_sim.current_location_id = get_location_id(node=agent_sim.start_node)
        if len(agent_sim.path_nodes) > 1: agent_sim.status = "MOVING"
        elif len(agent_sim.path_nodes) == 1 and agent_sim.start_node==agent_sim.goal_node: agent_sim.status = "ARRIVED"
        else: agent_sim.status = "NOT_STARTED"
        if not reserve_location(agent_sim.id, 0.0, agent_sim.current_location_id): agent_sim.status = "ERROR_START_BLOCKED"
        if G_osmnx_planning_graph.nodes.get(agent_sim.current_node):
            node_x = G_osmnx_planning_graph.nodes[agent_sim.current_node]['x']
            node_y = G_osmnx_planning_graph.nodes[agent_sim.current_node]['y']
            agent_artists[i].set_data([node_x], [node_y])
            agent_artists[i].set_color(route_colors_global[i % len(route_colors_global)])
            agent_artists[i].set_markersize(8 if agent_sim.status != "ARRIVED" else 10)
            agent_artists[i].set_markerfacecolor(route_colors_global[i % len(route_colors_global)] if agent_sim.status != "ARRIVED" else 'lime')
        else: agent_artists[i].set_data([],[])
    if ax_anim: ax_anim.set_title(f"Simülasyon Adımı: 0, Zaman: {current_anim_time:.1f}s")
    return agent_artists
def animate_frame_func(frame_num): # (Değişiklik yok)
    global current_anim_time, sim_agents_list_anim, G_osmnx_planning_graph, ani 
    active_agent_exists = False
    for i, agent_sim in enumerate(sim_agents_list_anim):
        if agent_sim.status == "MOVING":
            agent_sim.update_state(current_anim_time) 
            if agent_sim.status == "MOVING": active_agent_exists = True
        if agent_sim.on_edge:
            u, v = agent_sim.on_edge
            if G_osmnx_planning_graph.nodes.get(u) and G_osmnx_planning_graph.nodes.get(v):
                start_coords = np.array([G_osmnx_planning_graph.nodes[u]['x'], G_osmnx_planning_graph.nodes[u]['y']])
                end_coords = np.array([G_osmnx_planning_graph.nodes[v]['x'], G_osmnx_planning_graph.nodes[v]['y']])
                edge_data = G_osmnx_planning_graph.get_edge_data(u,v,0)
                edge_len = edge_data.get('length', 1.0) if edge_data else 1.0
                ratio = min(1.0, max(0.0, agent_sim.edge_progress / edge_len if edge_len > 0 else 0.0))
                curr_pos = start_coords + (end_coords - start_coords) * ratio
                agent_artists[i].set_data([curr_pos[0]], [curr_pos[1]])
            else: agent_artists[i].set_data([],[])
        else: 
            if G_osmnx_planning_graph.nodes.get(agent_sim.current_node):
                node_x = G_osmnx_planning_graph.nodes[agent_sim.current_node]['x']
                node_y = G_osmnx_planning_graph.nodes[agent_sim.current_node]['y']
                agent_artists[i].set_data([node_x], [node_y])
            else: agent_artists[i].set_data([],[])
        if agent_sim.status == "ARRIVED": agent_artists[i].set_markersize(10); agent_artists[i].set_markerfacecolor('lime')
        elif "ERROR" in agent_sim.status: agent_artists[i].set_markerfacecolor('black')
        else: agent_artists[i].set_markersize(8); agent_artists[i].set_markerfacecolor(route_colors_global[i % len(route_colors_global)])
    current_anim_time += DELTA_T
    if ax_anim: ax_anim.set_title(f"Simülasyon Adımı: {frame_num + 1}, Zaman: {current_anim_time:.1f}s")
    if not active_agent_exists and frame_num > MAX_SIM_TIME_STEPS // 10 : 
        if ani and ani.event_source: pass # ani.event_source.stop() 
    return agent_artists

# =============================================================================
# --- SUMO Rota Çevirici Fonksiyonları (Strateji B + C hibrit) ---
# =============================================================================
SUMO_EDGES_BY_NODES_CACHE = {} 
ORIG_ID_TO_SUMO_EDGES_MAP = {}

def build_sumo_edge_maps(sumo_net_obj):
    global SUMO_EDGES_BY_NODES_CACHE, ORIG_ID_TO_SUMO_EDGES_MAP
    if (SUMO_EDGES_BY_NODES_CACHE and ORIG_ID_TO_SUMO_EDGES_MAP) or not SUMOLIB_AVAILABLE or sumo_net_obj is None:
        return
    
    print("SUMO kenar haritaları (önbellekleri) oluşturuluyor...")
    node_pair_cache_count = 0
    orig_id_map_count = 0

    for edge in sumo_net_obj.getEdges(withInternal=True): # Kavşak içi kenarlar da dahil edilebilir.
        from_node_id = edge.getFromNode().getID()
        to_node_id = edge.getToNode().getID()
        edge_id = edge.getID()

        # Düğüm çifti -> kenar ID (ilk bulunanı al)
        if (from_node_id, to_node_id) not in SUMO_EDGES_BY_NODES_CACHE:
             SUMO_EDGES_BY_NODES_CACHE[(from_node_id, to_node_id)] = edge_id
             node_pair_cache_count +=1
        
        # origID -> [kenar nesneleri listesi]
        # SUMO Edge nesnesinden 'origId' parametresini almaya çalış
        # SUMO versiyonuna göre bu .param.get('origID') veya .getParameter('origID') olabilir.
        # Ya da doğrudan edge.get ursprünglD() gibi bir metod olabilir.
        # En güncel sumolib'de genellikle .getParameter("origID") kullanılır.
        orig_id_str = None
        try: # Farklı sumolib versiyonları için
            if hasattr(edge, 'getParameter') and edge.getParameter('origId'):
                orig_id_str = edge.getParameter('origId')
            elif hasattr(edge, 'param') and edge.param.get('origId'): # Eski versiyonlar
                 orig_id_str = edge.param.get('origId')
        except: # Bazen parametre sorgulama hata verebilir
            pass

        if not orig_id_str and '#' in edge_id: # origId yoksa, edge ID'den tahmin etmeye çalış
            potential_orig_id = edge_id.split('#')[0]
            if potential_orig_id.startswith("-"): potential_orig_id = potential_orig_id[1:]
            try: int(potential_orig_id); orig_id_str = potential_orig_id # Sayısal olup olmadığını kontrol et
            except ValueError: pass
        
        if orig_id_str:
            orig_id_str = str(orig_id_str) # Her zaman string olsun
            if orig_id_str not in ORIG_ID_TO_SUMO_EDGES_MAP:
                ORIG_ID_TO_SUMO_EDGES_MAP[orig_id_str] = []
            ORIG_ID_TO_SUMO_EDGES_MAP[orig_id_str].append(edge) # Kenar nesnesini sakla
            orig_id_map_count +=1
            
    print(f"{node_pair_cache_count} doğrudan düğüm-çifti eşlemesi, {len(ORIG_ID_TO_SUMO_EDGES_MAP)} origID için {orig_id_map_count} kenar girişi önbelleğe alındı.")


def get_ordered_sumo_edges_from_origid_candidates(osmnx_start_node_id, osmnx_end_node_id, 
                                                 candidate_sumo_edges, sumo_net_obj):
    """
    Verilen aday SUMO kenarları (aynı origID'ye sahip) içinden, osmnx_start_node'dan
    osmnx_end_node'a giden sıralı bir yol bulmaya çalışır.
    """
    sumo_start_node = sumo_net_obj.getNode(str(osmnx_start_node_id))
    sumo_end_node = sumo_net_obj.getNode(str(osmnx_end_node_id))

    if not sumo_start_node or not sumo_end_node or not candidate_sumo_edges:
        return None

    ordered_sumo_edge_ids = []
    current_sumo_node = sumo_start_node
    remaining_candidates = list(candidate_sumo_edges) # Kopya üzerinde çalış
    
    visited_edges_in_segment = set() # Bu segment için kullanılan kenarları takip et

    for _ in range(len(candidate_sumo_edges) + 2): # Olası maksimum adım sayısı + pay
        if current_sumo_node == sumo_end_node:
            break 

        found_next_edge_obj = None
        best_next_edge_obj = None
        
        # current_sumo_node'dan başlayan ve henüz kullanılmamış aday kenar ara
        for edge_obj in remaining_candidates:
            if edge_obj.getID() in visited_edges_in_segment: # Bu segmentte zaten kullanıldıysa atla
                continue
            if edge_obj.getFromNode() == current_sumo_node:
                # Eğer birden fazla seçenek varsa, hedef düğüme daha yakın olanı seçebiliriz (basit bir heuristik)
                # Şimdilik ilk bulduğumuzu alalım.
                best_next_edge_obj = edge_obj
                break 
        
        if best_next_edge_obj:
            ordered_sumo_edge_ids.append(best_next_edge_obj.getID())
            current_sumo_node = best_next_edge_obj.getToNode()
            visited_edges_in_segment.add(best_next_edge_obj.getID()) 
            # remaining_candidates.remove(best_next_edge_obj) # Dikkat: Bu, diğer segmentler için sorun olabilir, set kullanmak daha iyi.
        else:
            # print(f"  Sıralama Uyarısı: {current_sumo_node.getID()}'dan {sumo_end_node.getID()}'ye doğru aday kenar bulunamadı.")
            break # Daha fazla ilerleyemiyoruz

    if current_sumo_node == sumo_end_node and ordered_sumo_edge_ids:
        return ordered_sumo_edge_ids
    elif ordered_sumo_edge_ids: # Bir yol bulundu ama hedefe tam varamadı
        # print(f"  Sıralama Uyarısı: {osmnx_start_node_id}->{osmnx_end_node_id} için yol hedefe ulaşmadı. Bulunan: {ordered_sumo_edge_ids}")
        return None # Ya da kısmi yolu döndür? Şimdilik None.
    else:
        # print(f"  Sıralama Uyarısı: {osmnx_start_node_id}->{osmnx_end_node_id} için hiç sıralı yol bulunamadı.")
        return None


def get_sumo_edge_sequence_for_osmnx_segment_v_final(osmnx_node_u, osmnx_node_v, G_osmnx, sumo_net_obj):
    global SUMO_EDGES_BY_NODES_CACHE, ORIG_ID_TO_SUMO_EDGES_MAP
    if not SUMOLIB_AVAILABLE or sumo_net_obj is None: return None

    u_str, v_str = str(osmnx_node_u), str(osmnx_node_v)

    # Strateji 1: Doğrudan düğüm çifti -> kenar ID önbelleği
    direct_edge_id = SUMO_EDGES_BY_NODES_CACHE.get((u_str, v_str))
    if direct_edge_id:
        return [direct_edge_id]

    # Strateji 2: sumolib.getShortestPath(Node, Node) (En temel haliyle)
    try:
        from_j = sumo_net_obj.getNode(u_str)
        to_j = sumo_net_obj.getNode(v_str)
        if from_j and to_j:
            path_info = sumo_net_obj.getShortestPath(from_j, to_j)
            if path_info and path_info[0]:
                return [edge.getID() for edge in path_info[0]]
    except AttributeError as ae: # getShortestPath içindeki 'Node' object has no attribute 'getFunction' hatası için
        # print(f"  getShortestPath(Node,Node) AttributeError ({u_str}->{v_str}): {ae}. origID sıralama denenecek.")
        pass # Hata olursa origID yöntemine geç
    except Exception as e: # Diğer getShortestPath hataları
        # print(f"  getShortestPath(Node,Node) Genel Hata ({u_str}->{v_str}): {e}. origID sıralama denenecek.")
        pass

    # Strateji 3: OSMnx kenarının origID'sini kullanarak manuel sıralama
    try:
        osmnx_edge_data = G_osmnx.get_edge_data(osmnx_node_u, osmnx_node_v, 0)
        if not osmnx_edge_data: return None
        
        target_osmid = osmnx_edge_data.get('osmid')
        if not target_osmid: return None
        if isinstance(target_osmid, list): target_osmid = target_osmid[0]
        target_osmid_str = str(target_osmid)
        
        # Hem pozitif hem de negatif origID'yi kontrol etmeliyiz, çünkü osmnx'teki 'reversed' bilgisine
        # göre SUMO ID'si değişebilir.
        candidate_edges_positive = ORIG_ID_TO_SUMO_EDGES_MAP.get(target_osmid_str, [])
        candidate_edges_negative = ORIG_ID_TO_SUMO_EDGES_MAP.get("-" + target_osmid_str, [])
        
        # osmnx'in yönüyle (reversed flag) SUMO'daki olası yönü eşleştirmeye çalışalım
        is_osmnx_reversed = osmnx_edge_data.get('reversed', False)
        
        final_ordered_edges = None
        if not is_osmnx_reversed and candidate_edges_positive: # OSMnx yönü OSM way yönüyle aynı, pozitif ID'li adayları dene
            final_ordered_edges = get_ordered_sumo_edges_from_origid_candidates(osmnx_node_u, osmnx_node_v, candidate_edges_positive, sumo_net_obj)

        if not final_ordered_edges and is_osmnx_reversed and candidate_edges_negative: # OSMnx yönü ters, negatif ID'li adayları dene
            final_ordered_edges = get_ordered_sumo_edges_from_origid_candidates(osmnx_node_u, osmnx_node_v, candidate_edges_negative, sumo_net_obj)
        
        # Eğer yön bilgisiyle bulunamazsa, her iki yönü de dene (daha az kesin)
        if not final_ordered_edges and candidate_edges_positive:
            final_ordered_edges = get_ordered_sumo_edges_from_origid_candidates(osmnx_node_u, osmnx_node_v, candidate_edges_positive, sumo_net_obj)
        if not final_ordered_edges and candidate_edges_negative:
             final_ordered_edges = get_ordered_sumo_edges_from_origid_candidates(osmnx_node_u, osmnx_node_v, candidate_edges_negative, sumo_net_obj)

        if final_ordered_edges:
            # print(f"  origID sıralama ile bulundu: {u_str}->{v_str} (origID:{target_osmid_str}) => {final_ordered_edges}")
            return final_ordered_edges
            
    except Exception as e_orig:
        print(f"Hata (origID sıralama): {u_str}->{v_str} için: {e_orig}")

    # print(f"  SON ÇARE UYARI: {u_str}->{v_str} için SUMO kenar dizisi bulunamadı.")
    return None


def convert_to_sumo_routes_v_final(solution_paths, G_osmnx_graph, sumo_net_obj, output_file="generated_routes.rou.xml"):
    print(f"SUMO rota dosyası (v_final - hibrit) oluşturuluyor: {output_file}")
    routes_root = ET.Element("routes"); routes_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes_root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")
    max_s_overall = (DEFAULT_SPEED_LIMIT_KMPH * METER_PER_KM / SECONDS_PER_HOUR * 0.9)
    vtype = ET.SubElement(routes_root, "vType", id="car_cbs", accel=str(ACCELERATION), decel=str(DECELERATION), length="5", maxSpeed=f"{max_s_overall:.2f}", sigma="0.5", color="yellow", tau="1.5", carFollowModel="IDM")
    vehicle_count = 0
    for agent_id, path_tuples in solution_paths.items():
        if not path_tuples or len(path_tuples) < 2: continue
        depart_time = str(round(path_tuples[0][1], 2))
        osmnx_node_path = [item[0] for item in path_tuples]
        full_sumo_edge_route_list = []; valid_path_for_sumo = True
        for i in range(len(osmnx_node_path) - 1):
            u_osmnx, v_osmnx = osmnx_node_path[i], osmnx_node_path[i+1]
            sumo_edge_segment_ids = get_sumo_edge_sequence_for_osmnx_segment_v_final(u_osmnx, v_osmnx, G_osmnx_graph, sumo_net_obj)
            if sumo_edge_segment_ids and len(sumo_edge_segment_ids) > 0 : full_sumo_edge_route_list.extend(sumo_edge_segment_ids)
            else: print(f"KRİTİK UYARI (SUMO XML): Agent {agent_id} {u_osmnx}->{v_osmnx} için SUMO kenar dizisi yok. Rota atlanıyor."); valid_path_for_sumo = False; break 
        if valid_path_for_sumo and full_sumo_edge_route_list:
            vehicle_count +=1; agent_color_idx = (agent_id -1) % len(route_colors_global)
            color_str = "255,255,0"; mpl_color_name = route_colors_global[agent_color_idx]
            if mpl_color_name == 'r': color_str = "255,0,0"; 
            elif mpl_color_name == 'b': color_str = "0,0,255"
            elif mpl_color_name == 'darkorange': color_str = "255,140,0"; 
            elif mpl_color_name == 'fuchsia': color_str = "255,0,255"
            vehicle = ET.SubElement(routes_root, "vehicle", id=f"agent_{agent_id}", type="car_cbs", depart=depart_time, departLane="best", departSpeed="0", color=color_str) 
            ET.SubElement(vehicle, "route", edges=" ".join(full_sumo_edge_route_list))
    if vehicle_count == 0: print("UYARI: SUMO için geçerli Agent rotası oluşturulamadı.")
    try:
        rough_string = ET.tostring(routes_root, 'utf-8')
        reparsed = minidom.parseString(rough_string); pretty_xml_str = reparsed.toprettyxml(indent="  ")
        lines = pretty_xml_str.splitlines()
        final_xml_content = "\n".join(lines[1:]) if len(lines) > 1 and lines[0].strip().startswith("<?xml") else pretty_xml_str
        with open(output_file, "w", encoding="utf-8") as f: f.write(final_xml_content.strip())
        print(f"SUMO rota dosyası '{output_file}' başarıyla oluşturuldu ({vehicle_count} araç).")
    except Exception as e: print(f"HATA: XML yazılırken: {e}"); print(ET.tostring(routes_root, 'unicode'))

import matplotlib.pyplot as plt
import osmnx as ox # Eğer G_osmnx_planning_graph.nodes[node_id]['x'] için gerekliyse

# ... (Diğer görselleştirme fonksiyonlarınız ve ana kodunuz) ...

def plot_agent_coordinates_vs_time(final_timed_solution, G_osmnx_graph, agents_list, route_colors):
    """
    Her Agent için haritadaki X ve Y koordinatlarının zamana göre değişimini çizer.
    """
    if not final_timed_solution:
        print("plot_agent_coordinates_vs_time: Çözüm verisi bulunamadı.")
        return

    num_agents_in_solution = len(final_timed_solution)
    if num_agents_in_solution == 0:
        print("plot_agent_coordinates_vs_time: Çözümde Agent bulunamadı.")
        return

    # Her Agent için ayrı figürler oluşturabiliriz veya tek bir figürde alt grafikler
    # Şimdilik X ve Y koordinatları için ayrı ayrı iki figür oluşturalım.

    # --- X Koordinatı vs Zaman ---
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.title('Şekil 4a: Agentların X Koordinatları vs. Zaman', fontsize=14, fontweight='bold')
    plt.xlabel('Zaman (saniye)', fontsize=12)
    plt.ylabel('Harita X Koordinatı', fontsize=12)

    for agent_id_key in final_timed_solution:
        agent_id = int(agent_id_key) if isinstance(agent_id_key, str) and agent_id_key.isdigit() else agent_id_key
        path_tuples = final_timed_solution[agent_id_key]

        if not path_tuples:
            print(f"Uyarı (Koordinat Grafiği): Agent {agent_id} için yol verisi yok.")
            continue

        times = []
        x_coords = []

        for node_id, time_in_seconds in path_tuples:
            try:
                if node_id not in G_osmnx_graph.nodes:
                    print(f"Uyarı (Koordinat Grafiği): Agent {agent_id}, Düğüm {node_id} haritada bulunamadı.")
                    continue
                node_data = G_osmnx_graph.nodes[node_id]
                times.append(time_in_seconds)
                x_coords.append(node_data['x'])
            except KeyError:
                print(f"HATA (Koordinat Grafiği): Agent {agent_id}, Düğüm {node_id} için 'x' koordinatı bulunamadı.")
                continue
            except Exception as e:
                print(f"HATA (Koordinat Grafiği): Agent {agent_id}, Düğüm {node_id} işlenirken: {e}")
                continue

        if times and x_coords: # Sadece veri varsa çiz
            agent_obj = next((a for a in agents_list if a.id == agent_id), None)
            agent_label = f"Agent {agent_id}"
            if agent_obj:
                agent_label = f"Agent {agent_id} ({agent_obj.start_node} → {agent_obj.goal_node})"
            plt.plot(times, x_coords, marker='.', linestyle='-', label=agent_label, color=route_colors[(agent_id - 1) % len(route_colors)], linewidth=1.5, markersize=4)

    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # --- Y Koordinatı vs Zaman ---
    plt.figure(figsize=(14, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.title('Şekil 4b: Agentların Y Koordinatları vs. Zaman', fontsize=14, fontweight='bold')
    plt.xlabel('Zaman (saniye)', fontsize=12)
    plt.ylabel('Harita Y Koordinatı', fontsize=12)

    for agent_id_key in final_timed_solution:
        agent_id = int(agent_id_key) if isinstance(agent_id_key, str) and agent_id_key.isdigit() else agent_id_key
        path_tuples = final_timed_solution[agent_id_key]

        if not path_tuples:
            # Uyarı zaten yukarıda verildi.
            continue

        times = []
        y_coords = []

        for node_id, time_in_seconds in path_tuples:
            try:
                if node_id not in G_osmnx_graph.nodes:
                    # Uyarı zaten yukarıda verildi.
                    continue
                node_data = G_osmnx_graph.nodes[node_id]
                times.append(time_in_seconds)
                y_coords.append(node_data['y'])
            except KeyError:
                print(f"HATA (Koordinat Grafiği): Agent {agent_id}, Düğüm {node_id} için 'y' koordinatı bulunamadı.")
                continue
            except Exception as e:
                print(f"HATA (Koordinat Grafiği): Agent {agent_id}, Düğüm {node_id} işlenirken: {e}")
                continue
        
        if times and y_coords: # Sadece veri varsa çiz
            agent_obj = next((a for a in agents_list if a.id == agent_id), None)
            agent_label = f"Agent {agent_id}"
            if agent_obj:
                agent_label = f"Agent {agent_id} ({agent_obj.start_node} → {agent_obj.goal_node})"
            plt.plot(times, y_coords, marker='.', linestyle='-', label=agent_label, color=route_colors[(agent_id - 1) % len(route_colors)], linewidth=1.5, markersize=4)

    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

# ... (Mevcut GÖRSELLEŞTİRME FONKSİYONLARI bölümünüzün sonuna ekleyin) ...

# =============================================================================
# --- GÖRSELLEŞTİRME FONKSİYONLARI (Bu bölümü dosyanızın uygun bir yerine ekleyin) ---
# =============================================================================

def plot_space_time_graph(final_timed_solution, G_osmnx_graph, agents_list, route_colors):
    """ Grafik 3: Zaman-Mekân Grafiği """
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-whitegrid') # Daha hoş bir stil için

    agent_ids_in_solution = list(final_timed_solution.keys())
    colors = route_colors

    for i, agent_id in enumerate(agent_ids_in_solution):
        path_tuples = final_timed_solution[agent_id]
        if not path_tuples:
            print(f"Uyarı (Grafik 3): Agent {agent_id} için yol verisi yok.")
            continue

        times = []
        distances = []
        current_cumulative_distance = 0.0

        # İlk nokta (başlangıç)
        if not path_tuples[0] or len(path_tuples[0]) < 2:
            print(f"Uyarı (Grafik 3): Agent {agent_id} için başlangıç yol tupu geçersiz: {path_tuples[0]}")
            continue
        times.append(path_tuples[0][1]) # Başlangıç zamanı
        distances.append(0.0)
        previous_node = path_tuples[0][0]

        for k in range(1, len(path_tuples)):
            if not path_tuples[k] or len(path_tuples[k]) < 2:
                print(f"Uyarı (Grafik 3): Agent {agent_id}, adım {k} için yol tupu geçersiz: {path_tuples[k]}")
                continue # Bu adımı atla

            current_node_id, time_in_seconds = path_tuples[k]
            times.append(time_in_seconds)

            try:
                if previous_node == current_node_id: # Aynı düğümde bekleme
                    segment_length = 0.0
                elif G_osmnx_graph.has_edge(previous_node, current_node_id):
                    edge_data = G_osmnx_graph.get_edge_data(previous_node, current_node_id, 0)
                    segment_length = edge_data.get('length', 0.0)
                else: # Doğrudan kenar yoksa, bu beklenmedik bir durumdur.
                      # CBS çıktısının her zaman bağlı düğümler vermesi beklenir.
                      # Hata vermek veya kuş uçuşu mesafeyi kullanmak bir seçenek olabilir.
                      # Şimdilik, bu durumu bir uyarı ile geçelim ve mesafeyi 0 kabul edelim.
                    # print(f"Uyarı (Grafik 3): Agent {agent_id} için {previous_node} -> {current_node_id} arasında doğrudan kenar yok. Segment mesafesi 0 kabul ediliyor.")
                    # Alternatif: Kuş uçuşu mesafe (daha doğru ama yavaş olabilir ve harita projeksiyonuna bağlı)
                    node_prev_data = G_osmnx_graph.nodes[previous_node]
                    node_curr_data = G_osmnx_graph.nodes[current_node_id]
                    segment_length = ox.distance.great_circle_vec(node_prev_data['y'], node_prev_data['x'],
                                                                  node_curr_data['y'], node_curr_data['x'])


                current_cumulative_distance += segment_length
            except KeyError as e:
                print(f"HATA (Grafik 3): Mesafe hesaplanırken düğüm veya kenar bulunamadı: {previous_node} veya {current_node_id}. Hata: {e}")
                # Hata durumunda mesafeyi artırmayalım, bir önceki mesafede kalsın.
                # Bu, grafikte bir platoya neden olabilir veya hatalı bir çizgiye.
                # En son geçerli mesafeyi kullanmak daha iyi olabilir.
                if distances: distances.append(distances[-1]); # Son bilinen mesafeyi tekrarla
                else: distances.append(0.0) # Eğer ilk segmentte hata olursa
                previous_node = current_node_id # Bir sonraki adıma geçmek için güncelle
                continue # Bu segment için döngünün geri kalanını atla

            distances.append(current_cumulative_distance)
            previous_node = current_node_id

        agent_obj = next((a for a in agents_list if a.id == agent_id), None)
        agent_label = f"Agent {agent_id}"
        if agent_obj:
            agent_label = f"Agent {agent_id} ({agent_obj.start_node} → {agent_obj.goal_node})"

        plt.plot(times, distances, marker='o', linestyle='-', label=agent_label, color=colors[ (agent_id-1) % len(colors)], linewidth=2, markersize=5)

    plt.xlabel('Zaman (saniye)', fontsize=12)
    plt.ylabel('Kat Edilen Kümülatif Mesafe (metre)', fontsize=12)
    plt.title('Şekil 3: Agent Rotalarının Zaman-Mekân Grafiği', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()


def plot_cbs_conflict_resolution_example(G_osmnx_graph, initial_solution_timed, conflict_info, final_solution_timed, agents_list, route_colors):
    """ Grafik 1: CBS Çakışma Çözüm Örneği """
    fig, axes = plt.subplots(1, 2, figsize=(20, 9)) # Boyutu biraz artırdım
    fig.suptitle('Şekil 1: CBS Çakışma Çözümü Örneği', fontsize=16, fontweight='bold')
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = route_colors

    plot_kwargs = {'bgcolor': '#FFFFFF', 'node_size': 0, 'edge_color': 'silver', 'edge_linewidth': 0.8, 'show': False, 'close': False}

    # Alt Grafik (a): Çakışan Durum
    ax = axes[0]
    ox.plot_graph(G_osmnx_graph, ax=ax, **plot_kwargs)

    # initial_solution_timed: {agent_id: [(node, time), ...]} formatında olmalı
    if initial_solution_timed:
        for agent_id_key in initial_solution_timed: # agent_id string olabilir, int'e çevir
            agent_id = int(agent_id_key) if isinstance(agent_id_key, str) and agent_id_key.isdigit() else agent_id_key

            path_tuples = initial_solution_timed[agent_id_key]
            if path_tuples:
                path_nodes = [item[0] for item in path_tuples]
                agent_obj = next((a for a in agents_list if a.id == agent_id), None)
                start_node = agent_obj.start_node if agent_obj else path_nodes[0]
                end_node = agent_obj.goal_node if agent_obj else path_nodes[-1]
                current_color = colors[(agent_id-1) % len(colors)]

                if path_nodes: # Sadece rota varsa çiz
                    ox.plot_graph_route(G_osmnx_graph, path_nodes, route_color=current_color, route_linewidth=2.5, route_alpha=0.7,
                                        ax=ax, node_size=0, show=False, close=False)
                    if G_osmnx_graph.has_node(start_node):
                        ax.scatter(G_osmnx_graph.nodes[start_node]['x'], G_osmnx_graph.nodes[start_node]['y'],
                                   s=120, c=current_color, marker='o', edgecolors='black', linewidths=1, zorder=5, label=f'Agent{agent_id} Start')
                    if G_osmnx_graph.has_node(end_node):
                        ax.scatter(G_osmnx_graph.nodes[end_node]['x'], G_osmnx_graph.nodes[end_node]['y'],
                                   s=120, c=current_color, marker='X', edgecolors='black', linewidths=1, zorder=5, label=f'Agent{agent_id} Destination')

    if conflict_info and 'location' in conflict_info and isinstance(conflict_info['location'], int): # Düğüm çakışması
        conflict_node_id = conflict_info['location']
        if G_osmnx_graph.has_node(conflict_node_id):
            node_data = G_osmnx_graph.nodes[conflict_node_id]
            ax.scatter(node_data['x'], node_data['y'], color='red', s=300, zorder=6, marker='*',
                       label=f"Çakışma: N{conflict_node_id} @ T_idx{conflict_info['time_step']}")
        else:
            print(f"Uyarı (Grafik 1): Çakışma düğümü {conflict_node_id} haritada bulunamadı.")

    ax.set_title('(a) Başlangıç Rotaları ve Tespit Edilen Çakışma', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.set_facecolor('#F0F0F0') # Arka plan rengi

    # Alt Grafik (b): Çözülmüş Durum
    ax = axes[1]
    ox.plot_graph(G_osmnx_graph, ax=ax, **plot_kwargs)
    if final_solution_timed:
        for agent_id_key in final_solution_timed:
            agent_id = int(agent_id_key) if isinstance(agent_id_key, str) and agent_id_key.isdigit() else agent_id_key

            path_tuples = final_solution_timed[agent_id_key]
            if path_tuples:
                path_nodes = [item[0] for item in path_tuples]
                agent_obj = next((a for a in agents_list if a.id == agent_id), None)
                start_node = agent_obj.start_node if agent_obj else path_nodes[0]
                end_node = agent_obj.goal_node if agent_obj else path_nodes[-1]
                current_color = colors[(agent_id-1) % len(colors)]

                if path_nodes:
                    ox.plot_graph_route(G_osmnx_graph, path_nodes, route_color=current_color, route_linewidth=2.5, route_alpha=0.7,
                                        ax=ax, node_size=0, show=False, close=False)
                    if G_osmnx_graph.has_node(start_node):
                        ax.scatter(G_osmnx_graph.nodes[start_node]['x'], G_osmnx_graph.nodes[start_node]['y'],
                                   s=120, c=current_color, marker='o', edgecolors='black', linewidths=1, zorder=5, label=f'Agent{agent_id} Start')
                    if G_osmnx_graph.has_node(end_node):
                        ax.scatter(G_osmnx_graph.nodes[end_node]['x'], G_osmnx_graph.nodes[end_node]['y'],
                                   s=120, c=current_color, marker='X', edgecolors='black', linewidths=1, zorder=5, label=f'Agent{agent_id} Destination')

    ax.set_title('(b) CBS ile Çözülmüş Çakışmasız Rotalar', fontsize=12)
    ax.legend(fontsize=9, loc='best')
    ax.set_facecolor('#F0F0F0')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Suptitle ve eksen etiketleri için yer bırak


def plot_cbs_performance_scaling_single_run(num_agents_val, solve_time_val, total_cost_val):
    """ Grafik 2: CBS Performansının Ölçeklenmesi (Tek Çalıştırma İçin Basitleştirilmiş) """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    agent_counts = [num_agents_val] # Tek bir değer olduğu için liste içinde
    solve_times = [solve_time_val] if solve_time_val is not None and solve_time_val != -1 else []
    total_costs = [total_cost_val] if total_cost_val is not None and total_cost_val != -1 else []

    plotted_time = False
    plotted_cost = False

    if solve_times:
        color_time = 'crimson' # Daha belirgin bir kırmızı
        ax1.set_xlabel('Agent Sayısı', fontsize=12)
        ax1.set_ylabel('Çözüm Süresi (saniye)', color=color_time, fontsize=12)
        ax1.plot(agent_counts, solve_times, color=color_time, marker='o', markersize=8, linestyle='None', label='Çözüm Süresi')
        ax1.tick_params(axis='y', labelcolor=color_time, labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        plotted_time = True
    else: # Sadece maliyet varsa, sol y eksenini maliyet için kullan
        color_cost = 'mediumblue' # Daha belirgin bir mavi
        ax1.set_xlabel('Agent Sayısı', fontsize=12)
        ax1.set_ylabel('Toplam Çözüm Maliyeti (saniye)', color=color_cost, fontsize=12) # Maliyet birimi saniye ise
        if total_costs: # Sadece total_costs doluysa çiz
             ax1.plot(agent_counts, total_costs, color=color_cost, marker='s', markersize=8, linestyle='None', label='Toplam Maliyet')
        ax1.tick_params(axis='y', labelcolor=color_cost, labelsize=10)
        ax1.tick_params(axis='x', labelsize=10)
        plotted_cost = True if total_costs else False


    if plotted_time and total_costs: # Hem süre hem maliyet varsa (ve süre zaten çizildiyse), ikinci y ekseni
        ax2 = ax1.twinx()
        color_cost = 'mediumblue'
        ax2.set_ylabel('Toplam Çözüm Maliyeti (saniye)', color=color_cost, fontsize=12) # Maliyet birimi
        ax2.plot(agent_counts, total_costs, color=color_cost, marker='s', markersize=8, linestyle='None', label='Toplam Maliyet')
        ax2.tick_params(axis='y', labelcolor=color_cost, labelsize=10)
        plotted_cost = True

    # X ekseninde sadece mevcut Agent sayısını göster
    if agent_counts:
        ax1.set_xticks(agent_counts)
        ax1.set_xticklabels([str(ac) for ac in agent_counts])


    plt.title(f'Şekil 2: CBS Performansı ({num_agents_val} Agent için Tek Çalıştırma)', fontsize=14, fontweight='bold')

    # Legendları birleştirme
    handles, labels = [], []
    if plotted_time:
        h, l = ax1.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    if plotted_cost and 'ax2' in locals(): # Eğer ax2 oluşturulduysa
        h, l = ax2.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)
    elif plotted_cost and not plotted_time: # Sadece maliyet ax1'de çizildiyse
        h, l = ax1.get_legend_handles_labels()
        handles.extend(h); labels.extend(l)

    if handles:
      fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=10, frameon=True, facecolor='white', edgecolor='gray')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.90]) # Legend için yer bırak

import matplotlib.pyplot as plt
import osmnx as ox
import numpy as np
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = False
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("Uyarı: 'adjustText' kütüphanesi bulunamadı. Zaman etiketleri üst üste binebilir. Kurulum için: pip install adjustText")

def plot_routes_on_map_with_time_markers(G_osmnx_graph, final_timed_solution, agents_list, route_colors,
                                         time_interval=20, show_time_labels=True, use_adjust_text=True):
    if not final_timed_solution:
        print("plot_routes_on_map_with_time_markers: Çözüm verisi bulunamadı.")
        return

    # Figür boyutunu biraz daha büyük tutabiliriz, yazılar için yer açılsın.
    fig, ax = ox.plot_graph(G_osmnx_graph, show=False, close=False, bgcolor='#FFFFFF',
                            node_size=0, edge_color='silver', edge_linewidth=0.6, figsize=(18, 18)) # figsize'ı artırdım
    
    plt.title(f' Agent Routes at {time_interval}s Time Intervals', fontsize=18, fontweight='bold') # Başlık fontunu da büyüttüm

    max_time_overall = 0
    # ... (max_time_overall ve diğer değişkenlerin hesaplanması - önceki cevaplardaki gibi) ...
    for agent_id_key in final_timed_solution:
        path_tuples = final_timed_solution[agent_id_key]
        if path_tuples and path_tuples[-1] and path_tuples[-1][1] is not None:
            max_time_overall = max(max_time_overall, path_tuples[-1][1])

    if max_time_overall == 0 and final_timed_solution :
        if any(final_timed_solution.values()):
             print("Uyarı: Grafikte Time-Progressaretçisi oluşturulamadı, maksimum zaman 0 veya rotalar boş.")

    time_markers_to_plot = np.arange(time_interval, max_time_overall + time_interval, time_interval)
    legend_handles_labels = {}
    text_labels_for_adjust = []

    min_x, min_y, max_x, max_y = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
    x_range = max_x - min_x
    y_range = max_y - min_y
    # Dinamik ofsetleri biraz daha büyük yapabiliriz veya sabit bir değer deneyebiliriz
    dynamic_offset_x = x_range * 0.001 
    dynamic_offset_y = y_range * 0.001


    for agent_idx, agent_id_key in enumerate(final_timed_solution):
        # ... (agent_id, path_tuples, current_color, agent_obj, agent_label_base tanımlamaları - önceki cevaplardaki gibi) ...
        try:
            agent_id = int(agent_id_key)
        except (ValueError, TypeError):
             if isinstance(agent_id_key, int):
                 agent_id = agent_id_key
             else:
                print(f"Uyarı: Agent ID '{agent_id_key}' beklenmedik formatta. Atlanıyor olabilir.")
                continue

        path_tuples = final_timed_solution.get(agent_id_key)
        if not path_tuples: continue

        path_nodes_only = [item[0] for item in path_tuples if item and len(item)>0]
        if not path_nodes_only: continue

        current_color = route_colors[(agent_id - 1) % len(route_colors)]
        agent_obj = next((a for a in agents_list if a.id == agent_id), None)
        agent_label_base = f"Agent {agent_id}"
        if agent_obj:
            agent_label_base = ''#f"Agent {agent_id} ({getattr(agent_obj, 'start_node', 'N/A')}→{getattr(agent_obj, 'goal_node', 'N/A')})"

        # Ana rotayı, başlangıç ve hedef noktalarını çizme...
        # ... (Bu kısımlar önceki cevaplardaki gibi kalabilir) ...
        if path_nodes_only:
            route_plot_elements = ox.plot_graph_route(G_osmnx_graph, path_nodes_only, route_color=current_color,
                                                      route_linewidth=2, route_alpha=0.6, ax=ax, show=False, close=False)
            route_label_key = f"{agent_label_base} Rota"
            if route_label_key not in legend_handles_labels and route_plot_elements:
                 legend_handles_labels[route_label_key] = route_plot_elements[0]

            start_node_id = path_nodes_only[0]
            end_node_id = path_nodes_only[-1]
            if G_osmnx_graph.has_node(start_node_id):
                start_scatter = ax.scatter(G_osmnx_graph.nodes[start_node_id]['x'], G_osmnx_graph.nodes[start_node_id]['y'],
                                           s=100, c=current_color, marker='o', edgecolors='black', linewidths=0.8, zorder=4)
                start_label_key = f"{agent_label_base} Start"
                if start_label_key not in legend_handles_labels:
                    legend_handles_labels[start_label_key] = start_scatter
            if G_osmnx_graph.has_node(end_node_id):
                end_scatter = ax.scatter(G_osmnx_graph.nodes[end_node_id]['x'], G_osmnx_graph.nodes[end_node_id]['y'],
                                         s=100, c=current_color, marker='X', edgecolors='black', linewidths=0.8, zorder=4)
                end_label_key = f"{agent_label_base} Destination"
                if end_label_key not in legend_handles_labels:
                    legend_handles_labels[end_label_key] = end_scatter


        plotted_time_markers_for_agent_legend = False
        for t_idx, t_marker in enumerate(time_markers_to_plot):
            interp_x, interp_y = None, None
            # ... (interpolasyonla interp_x, interp_y bulma - önceki cevaplardaki gibi) ...
            if not path_tuples or len(path_tuples) < 2: continue
            for i in range(len(path_tuples) - 1):
                if not path_tuples[i] or len(path_tuples[i]) < 2 or \
                   not path_tuples[i+1] or len(path_tuples[i+1]) < 2:
                    continue
                node_id, time_val = path_tuples[i]
                next_node_id, next_time_val = path_tuples[i+1]
                if time_val is None or next_time_val is None: continue
                if time_val <= t_marker <= next_time_val:
                    if abs(next_time_val - time_val) > 1e-6:
                        ratio = (t_marker - time_val) / (next_time_val - time_val)
                    else: ratio = 0
                    if not (G_osmnx_graph.has_node(node_id) and G_osmnx_graph.has_node(next_node_id)): continue
                    try:
                        x1, y1 = G_osmnx_graph.nodes[node_id]['x'], G_osmnx_graph.nodes[node_id]['y']
                        x2, y2 = G_osmnx_graph.nodes[next_node_id]['x'], G_osmnx_graph.nodes[next_node_id]['y']
                        interp_x = x1 + ratio * (x2 - x1)
                        interp_y = y1 + ratio * (y2 - y1)
                        break 
                    except KeyError as e:
                        print(f"Düğüm koordinatı alınırken hata: {e}. Düğüm ID: {node_id} veya {next_node_id}")
                        continue
            else:
                if path_tuples and path_tuples[-1] and len(path_tuples[-1]) >=2 and \
                   path_tuples[-1][1] is not None and t_marker >= path_tuples[-1][1] and \
                   G_osmnx_graph.has_node(path_tuples[-1][0]):
                    try:
                        interp_x = G_osmnx_graph.nodes[path_tuples[-1][0]]['x']
                        interp_y = G_osmnx_graph.nodes[path_tuples[-1][0]]['y']
                    except KeyError as e:
                        print(f"Son düğüm koordinatı alınırken hata: {e}. Düğüm ID: {path_tuples[-1][0]}")
                        interp_x, interp_y = None, None
                elif path_tuples and path_tuples[0] and len(path_tuples[0]) >=2 and \
                     path_tuples[0][1] is not None and t_marker < path_tuples[0][1]:
                    continue

            if interp_x is not None and interp_y is not None:
                marker_legend_label = None
                # ... (işaretçi Legendı ekleme) ...
                if not plotted_time_markers_for_agent_legend:
                    marker_legend_label = f'Agent {agent_id} Time-Progress.'
                    plotted_time_markers_for_agent_legend = True
                
                time_marker_scatter = ax.scatter(interp_x, interp_y, s=70, c=current_color, marker='D', # İşaretçi boyutunu biraz artırdım
                                                 edgecolors='white', linewidths=0.8, zorder=5, alpha=0.95, 
                                                 label=marker_legend_label if marker_legend_label else "_nolegend_")
                
                if marker_legend_label and marker_legend_label not in legend_handles_labels:
                    legend_handles_labels[marker_legend_label] = time_marker_scatter

                if show_time_labels:
                    offset_sign_x = 1 if (agent_idx + t_idx) % 4 < 2 else -1
                    offset_sign_y = 1 if (agent_idx + t_idx + 1) % 4 < 2 else -1
                    text_x = interp_x + offset_sign_x * dynamic_offset_x
                    text_y = interp_y + offset_sign_y * dynamic_offset_y
                    
                    time_label_obj = ax.text(text_x, text_y, f"t={t_marker:.0f}", color='black', 
                                           # YAZI TİPİ BOYUTUNU ARTIRDIM:
                                           fontsize=10, # Önceki 5 idi, şimdi 7 veya 8 deneyebilirsiniz
                                           fontweight='normal', 
                                           ha='center' if offset_sign_x == 0 else ('left' if offset_sign_x > 0 else 'right'),
                                           va='center' if offset_sign_y == 0 else ('bottom' if offset_sign_y > 0 else 'top'),
                                           bbox=dict(boxstyle='round,pad=0.1', fc=current_color, ec='none', alpha=0.6)) # pad'i biraz artırdım
                    text_labels_for_adjust.append(time_label_obj)

    if ADJUST_TEXT_AVAILABLE and use_adjust_text and text_labels_for_adjust:
        print("adjustText ile zaman etiketleri ayarlanıyor...")
        adjust_text(text_labels_for_adjust, 
                    # expand_points=(1.2,1.2), expand_text=(1.2,1.2), # Bu değerlerle oynayabilirsiniz
                    arrowprops=dict(arrowstyle="-", color='gray', lw=0.5, alpha=0.6))

    if legend_handles_labels:
        # Legend YAZI TİPİ BOYUTLARINI ARTIRDIM:
        ax.legend(legend_handles_labels.values(), legend_handles_labels.keys(), 
                  fontsize=10,  # Önceki 6 idi, şimdi 8 veya 'small'
                  loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., 
                  frameon=True, facecolor='white', framealpha=0.8, 
                  title="Legend", title_fontsize=9) # Önceki 7 idi, şimdi 9 veya 'small'
    
    # Legend için sağda daha fazla yer bırakmak ve genel yerleşimi ayarlamak için:
    fig.subplots_adjust(left=0.05, right=0.75, top=0.92, bottom=0.05) # right değerini biraz daha küçülttüm, top ve bottom ayarlandı
    # plt.tight_layout(rect=[0, 0, 0.78, 0.95]) # rect parametrelerini ayarlayarak başlık ve Legend için yer bırakın
    # tight_layout bazen subplots_adjust ile çakışabilir, birini veya diğerini kullanın veya dikkatli ayarlayın.
# =============================================================================
# --- GÖRSELLEŞTİRME FONKSİYONLARI SONU ---
# =============================================================================


# =============================================================================
# --- Ana Program Bloğu ---
# =============================================================================
if __name__ == "__main__":
    start_time_main = time.time()

    osm_file_path = "map.osm"
    sumo_net_file_path = "map.net.xml" # map.osm'den üretilmiş SUMO ağı

    # --- 1. Harita ve Ağ Yükleme ---
    try:
        print(f"OSM dosyasından ağ yükleniyor: {osm_file_path}")
        G_raw = ox.graph_from_xml(osm_file_path, simplify=False, retain_all=True)
        G_proj = ox.project_graph(G_raw)
        G_osmnx_planning_graph = ox.simplify_graph(G_proj.copy())
        fig, ax = ox.plot_graph(G_raw, node_size=5, edge_linewidth=0.5, bgcolor='white')
        fig, ax = ox.plot_graph(G_osmnx_planning_graph,
                        node_size=5,
                        edge_linewidth=0.5,
                        bgcolor='white',
                        node_color='blue',
                        edge_color='gray',
                        save=True,
                        filepath='graph_raw.png',
                        dpi=300)
        fig, ax = ox.plot_graph(G_raw,
                        node_size=5,
                        edge_linewidth=0.5,
                        bgcolor='white',
                        node_color='blue',
                        edge_color='gray',
                        save=True,
                        filepath='graph_raw.png',
                        dpi=300)

        if not G_osmnx_planning_graph.is_directed():
            G_osmnx_planning_graph = ox.get_digraph(G_osmnx_planning_graph.copy())
            
        print(f"Rota planlama grafiği oluşturuldu. Düğümler: {G_osmnx_planning_graph.number_of_nodes()}, Kenarlar: {G_osmnx_planning_graph.number_of_edges()}, Yönlü: {G_osmnx_planning_graph.is_directed()}")
    except Exception as e:
        print(f"HATA: Harita yüklenirken/işlenirken: {e}"); exit()
        
    if SUMOLIB_AVAILABLE:
        try:
            sumo_net_obj_global = sumolib.net.readNet(sumo_net_file_path, withInternal=True)
            print(f"SUMO ağı '{sumo_net_file_path}' başarıyla yüklendi.")
            build_sumo_edge_maps(sumo_net_obj_global)
        except FileNotFoundError:
            print(f"UYARI: SUMO ağı '{sumo_net_file_path}' bulunamadı!"); sumo_net_obj_global = None
        except Exception as e:
            print(f"UYARI: SUMO ağı '{sumo_net_file_path}' yüklenirken hata: {e}."); sumo_net_obj_global = None
    else:
        print("UYARI: sumolib yüklenemediği için SUMO kenar ID'leri doğru bulunamayabilir.")
        sumo_net_obj_global = None # sumolib yoksa bu nesne None olmalı

    # --- 2. Agentların Oluşturulması ---
    try:
        components = list(nx.strongly_connected_components(G_osmnx_planning_graph))
        if not components: components = list(nx.weakly_connected_components(G_osmnx_planning_graph))
        if not components: raise ValueError("Bağlantılı bileşen bulunamadı.")
        largest_component_nodes = max(components, key=len)
        valid_nodes = list(largest_component_nodes)
        print(f"En büyük bağlantılı bileşende {len(valid_nodes)} düğüm bulundu.")
    except Exception as e:
        print(f"HATA: Bağlantılı bileşenler bulunurken: {e}"); exit()

    num_agents = 5 # Agent sayısını buradan ayarlayabilirsiniz
    if len(valid_nodes) < num_agents * 2:
        print(f"HATA: Yeterli benzersiz düğüm yok ({len(valid_nodes)} < {num_agents*2}). Agent sayısını azaltın veya daha büyük bir harita kullanın."); exit()

    selected_nodes = random.sample(valid_nodes, num_agents * 2)
    agents_for_cbs = []
    for i in range(num_agents):
        start_n, goal_n = selected_nodes[i], selected_nodes[i + num_agents]
        if start_n == goal_n: # Başlangıç ve hedef aynı olmasın
            temp_goals = [n for n in valid_nodes if n != start_n and n not in [s[1] for s in selected_nodes[num_agents:] if s[0] != start_n]] # Diğer hedeflerden de farklı olsun
            if not temp_goals: # Eğer hiç alternatif kalmazsa (çok küçük harita/çok Agent)
                 # Farklı bir başlangıç noktası seçmeyi deneyebiliriz veya hata verebiliriz.
                 # Şimdilik basitçe farklı bir düğüm seçiyoruz.
                 available_nodes_for_goal = [n for n in valid_nodes if n != start_n]
                 if not available_nodes_for_goal: print(f"HATA: Agent {i+1} için {start_n} dışında hedef düğüm bulunamadı."); exit()
                 goal_n = random.choice(available_nodes_for_goal)
            else:
                 goal_n = random.choice(temp_goals)

        agents_for_cbs.append(Agent(i + 1, start_n, goal_n, G_osmnx_planning_graph))
    print("\n--- Agentlar Oluşturuldu (CBS için) ---")
    for agent in agents_for_cbs: print(agent)

    # --- 3. Conflict-Based Search (CBS) Çalıştırması ---
    print("\nCBS algoritması çalıştırılıyor...")
    cbs_start_time_val = time.time()

    # run_cbs fonksiyonunuzun güncellenmiş halini çağırın:
    # final_timed_solution, initial_solution_for_graph1, first_conflict_for_graph1, cbs_total_cost_val = run_cbs(agents_for_cbs, G_osmnx_planning_graph)
    # Şimdilik, run_cbs'in sadece final_timed_solution döndürdüğünü varsayarak devam ediyorum.
    # Grafik 1 ve 2'nin tam çalışması için run_cbs'i güncellemeniz GEREKİR.
    final_timed_solution = run_cbs(agents_for_cbs, G_osmnx_planning_graph) # BU SATIRI GÜNCELLEMELİSİNİZ!

    cbs_solve_time_val = time.time() - cbs_start_time_val
    print(f"CBS çalışma süresi: {cbs_solve_time_val:.3f} saniye.")

    # Grafik 1 ve 2 için verileri manuel olarak (veya geçici olarak) ayarlıyoruz.
    # run_cbs güncellendiğinde bu kısımlar otomatik dolacak.
    initial_solution_for_graph1 = None # run_cbs'ten gelmeli
    first_conflict_for_graph1 = None   # run_cbs'ten gelmeli
    cbs_total_cost_val = float('inf')  # run_cbs'ten gelmeli veya final_timed_solution'dan hesaplanmalı

    if final_timed_solution:
        # cbs_total_cost_val'ı final_timed_solution'dan hesapla (eğer run_cbs döndürmüyorsa)
        temp_cost = 0
        valid_solution_for_cost = True
        for path_data in final_timed_solution.values():
            if path_data and path_data[-1][1] is not None:
                temp_cost += path_data[-1][1]
            else:
                valid_solution_for_cost = False; break
        if valid_solution_for_cost: cbs_total_cost_val = temp_cost

        # Grafik 1 için DUMMY VERİ (run_cbs güncellenene kadar)
        # Bu veriler SADECE Grafik 1'in hata vermeden çalışması içindir, ANLAMLI DEĞİLDİR.
        if num_agents > 0 and not initial_solution_for_graph1:
            # Basitçe ilk Agentın başlangıç çözümünü kopyala (anlamsız ama hata vermez)
            # Gerçekçi olması için run_cbs'ten kısıtsız A* çözümlerini almalısınız.
            first_agent_id = agents_for_cbs[0].id
            if first_agent_id in final_timed_solution:
                 initial_solution_for_graph1 = {first_agent_id: final_timed_solution[first_agent_id]}
                 if num_agents > 1: # İkinci Agent için de ekle
                     second_agent_id = agents_for_cbs[1].id
                     if second_agent_id in final_timed_solution:
                        initial_solution_for_graph1[second_agent_id] = final_timed_solution[second_agent_id]


        if num_agents > 1 and not first_conflict_for_graph1 and initial_solution_for_graph1:
            # Çok basit bir dummy çakışma (ilk iki Agentın başlangıç düğümünde T=0'da gibi)
            # Bu da ANLAMSIZDIR. Gerçek ilk çakışmayı run_cbs'ten almalısınız.
            try:
                node_for_conflict = initial_solution_for_graph1[agents_for_cbs[0].id][0][0]
                first_conflict_for_graph1 = {'agent1': agents_for_cbs[0].id, 'agent2': agents_for_cbs[1].id, 'location': node_for_conflict, 'time_step': 0}
            except: pass # Dummy veri oluşturma başarısız olursa sorun değil.

        if not initial_solution_for_graph1 or not first_conflict_for_graph1 :
            print("\nUYARI: Grafik 1 için 'initial_solution_for_graph1' ve/veya 'first_conflict_for_graph1' verileri eksik veya dummy. Lütfen run_cbs fonksiyonunu güncelleyin.")


    # --- 4. Sonuçların İşlenmesi ve SUMO/Animasyon Hazırlığı ---
    agent_paths_viz = {}
    cbs_paths_found = False
    if final_timed_solution:
        cbs_paths_found = True
        all_agents_have_paths = True
        for agent_obj in agents_for_cbs:
            agent_id = agent_obj.id
            if agent_id in final_timed_solution and final_timed_solution[agent_id]:
                path_tuples = final_timed_solution[agent_id]
                path_nodes_only = [item[0] for item in path_tuples]
                agent_obj.set_path(path_nodes_only)
                agent_paths_viz[agent_id] = path_nodes_only
            else:
                print(f"Uyarı: Agent {agent_id} için CBS çözümü bulunamadı veya boş.")
                all_agents_have_paths = False # Eğer bir Agent için yol yoksa animasyon/sumo için sorun olabilir.
                # cbs_paths_found True kalabilir, ama bazı Agentların yolu olmayabilir.
        if not all_agents_have_paths and not any(agent_paths_viz.values()): # Hiçbir Agent için yol yoksa
            cbs_paths_found = False

    else:
        print("\nCBS çakışmasız bir çözüm bulamadı.")

    # --- 5. SUMO Rota Dosyası Oluşturma ---
    if cbs_paths_found and SUMOLIB_AVAILABLE and sumo_net_obj_global:
        print("\nCBS çözüm bulduğu için SUMO rota dosyası oluşturuluyor...")
        convert_to_sumo_routes_v_final(final_timed_solution,
                                       G_osmnx_planning_graph,
                                       sumo_net_obj_global,
                                       output_file="generated_routes.rou.xml")
    elif not cbs_paths_found:
        print("\nCBS çözüm bulamadığı için SUMO rota dosyası oluşturulmuyor.")
    elif not SUMOLIB_AVAILABLE or not sumo_net_obj_global:
        print("\nsumolib veya SUMO ağı yüklenemediği için SUMO rota dosyası oluşturulmuyor.")

    # --- 6. Animasyon ---
    # fig_anim ve ani globalde tanımlı, burada kullanılacak.
    #global fig_anim, ani # Eğer animasyon fonksiyonları bunları global olarak bekliyorsa
    show_animation = True # Animasyonu göstermek isteyip istemediğinizi buradan kontrol edin

    if cbs_paths_found and any(agent_paths_viz.values()) and show_animation:
        print("\n--- Rotalar Hazır: Animasyon Başlatılıyor ---")
        try:
            fig_anim, ax_anim = ox.plot_graph(G_osmnx_planning_graph, show=False, close=False, bgcolor='k',
                                              node_size=0, edge_color='gray', edge_linewidth=0.3, figsize=(12,12))
            # Statik rotaları çiz
            for i_viz, agent_id_viz_key in enumerate(agent_paths_viz.keys()):
                path_nodes_viz = agent_paths_viz[agent_id_viz_key]
                if path_nodes_viz:
                    agent_id_viz = int(agent_id_viz_key) if isinstance(agent_id_viz_key, str) and agent_id_viz_key.isdigit() else agent_id_viz_key
                    color_viz = route_colors_global[ (agent_id_viz -1) % len(route_colors_global)]
                    agent_obj_viz = next((a for a in agents_for_cbs if a.id == agent_id_viz), None)
                    if agent_obj_viz:
                        start_n_viz = agent_obj_viz.start_node
                        end_n_viz = agent_obj_viz.goal_node
                        ox.plot_graph_route(G_osmnx_planning_graph, path_nodes_viz, route_color=color_viz, route_alpha=0.4,
                                            route_linewidth=3, ax=ax_anim, node_size=0, show=False, close=False)
                        if G_osmnx_planning_graph.has_node(start_n_viz) and G_osmnx_planning_graph.has_node(end_n_viz):
                            ax_anim.scatter(G_osmnx_planning_graph.nodes[start_n_viz]['x'], G_osmnx_planning_graph.nodes[start_n_viz]['y'],
                                            s=120, c=color_viz, marker='o', zorder=5, label=f'A{agent_id_viz} Başl.', edgecolors='w')
                            ax_anim.scatter(G_osmnx_planning_graph.nodes[end_n_viz]['x'], G_osmnx_planning_graph.nodes[end_n_viz]['y'],
                                            s=120, c=color_viz, marker='X', zorder=5, label=f'A{agent_id_viz} Hedef', edgecolors='w')
            # Animasyon için agent_artists ve sim_agents_list_anim hazırlanması
            agent_artists.clear() # Önceki çalıştırmalardan kalıntıları temizle
            for i in range(len(agents_for_cbs)): # agents_for_cbs listesindeki tüm Agentlar için artist oluştur
                artist, = ax_anim.plot([], [], marker='o', markersize=7, zorder=10)
                agent_artists.append(artist)

            sim_agents_list_anim.clear() # Önceki çalıştırmalardan kalıntıları temizle
            for original_agent in agents_for_cbs:
                 # Sadece CBS'de yolu bulunan ve animasyon için hazırlanan Agentları ekle
                 if original_agent.id in agent_paths_viz and agent_paths_viz[original_agent.id]:
                    sim_agent = Agent(original_agent.id, original_agent.start_node, original_agent.goal_node, original_agent.G)
                    sim_agent.set_path(original_agent.path_nodes) # original_agent.path_nodes CBS sonrası güncellenmiş olmalı
                    sim_agents_list_anim.append(sim_agent)

            if sim_agents_list_anim: # Eğer simüle edilecek Agent varsa animasyonu başlat
                max_arrival_time = 0
                if final_timed_solution:
                    for agent_id_key in final_timed_solution:
                        path = final_timed_solution[agent_id_key]
                        if path and path[-1][1] is not None : max_arrival_time = max(max_arrival_time, path[-1][1])

                num_animation_frames = MAX_SIM_TIME_STEPS
                if max_arrival_time > 0:
                    num_animation_frames = int(np.ceil(max_arrival_time / DELTA_T)) + 20 # Biraz pay bırak
                    num_animation_frames = min(num_animation_frames, MAX_SIM_TIME_STEPS * 3)
                print(f"Animasyon kare sayısı: {num_animation_frames}")

                ani = animation.FuncAnimation(fig_anim, animate_frame_func, init_func=init_animation_func,
                                              frames=num_animation_frames, interval=100, blit=False, repeat=False) # interval'i düşürdüm
                if ax_anim: ax_anim.legend(loc='upper right', facecolor='w', framealpha=0.7, fontsize='small')
                if fig_anim: fig_anim.suptitle("Çoklu Agent Rota Simülasyonu (CBS ile)", fontsize=16, fontweight='bold')

                plt.show() # Animasyonu göster
            else:
                print("Animasyon için simüle edilecek Agent bulunamadı (CBS yolları eksik olabilir).")
                fig_anim = None # fig_anim'i sıfırla ki diğer grafikler için plt.show() çalışsın
        except Exception as e_anim:
            print(f"HATA: Animasyon oluşturulurken: {e_anim}")
            import traceback
            traceback.print_exc()
            fig_anim = None # Hata olursa fig_anim'i sıfırla
            # plt.close('all') # Açık figürleri kapatıp diğerlerine geçebiliriz

    elif cbs_paths_found:
        print("\nCBS rotaları bulundu ancak animasyon için uygun yol(lar) yok veya animasyon kapalı.")
        fig_anim = None # fig_anim'i sıfırla
    else:
        print("\nCBS çözüm bulamadığı için animasyon oluşturulmuyor.")
        fig_anim = None # fig_anim'i sıfırla


    # --- 7. Sonuç Grafikleri ---
    if cbs_paths_found and G_osmnx_planning_graph is not None:
        print("\n--- Sonuç Grafikleri Oluşturuluyor ---")
        figures_created = False

        # Grafik 3: Zaman-Mekân Grafiği
        try:
            print("Grafik 3: Zaman-Mekân Grafiği oluşturuluyor...")
            plot_space_time_graph(final_timed_solution, G_osmnx_planning_graph, agents_for_cbs, route_colors_global)
            figures_created = True
        except Exception as e:
            print(f"HATA: Grafik 3 (Zaman-Mekân) oluşturulurken: {e}")
            import traceback; traceback.print_exc()
            # YENİ GRAFİK: Agent Koordinatları vs. Zaman
        try:
            print("Grafik 5: Harita Üzerinde Time-Progressaretli Rotalar oluşturuluyor...")
            plot_routes_on_map_with_time_markers(G_osmnx_planning_graph, final_timed_solution, agents_for_cbs, route_colors_global, time_interval=20) # time_interval'i ayarlayabilirsiniz
            figures_created = True
        except Exception as e:
            print(f"HATA: Grafik 5 (Harita Üzerinde Time-Progressaretleri) oluşturulurken: {e}")
            import traceback; traceback.print_exc()

        # Grafik 1: CBS Çakışma Çözüm Örneği
        # Bu grafiğin doğru çalışması için initial_solution_for_graph1 ve first_conflict_for_graph1'in
        # run_cbs'ten doğru ve anlamlı bir şekilde doldurulması GEREKİR.
        if initial_solution_for_graph1 and first_conflict_for_graph1:
            try:
                print("Grafik 1: CBS Çakışma Çözüm Örneği oluşturuluyor...")
                plot_cbs_conflict_resolution_example(
                    G_osmnx_planning_graph,
                    initial_solution_for_graph1,
                    first_conflict_for_graph1,
                    final_timed_solution, # Nihai çözüm de Grafik 1 için önemli
                    agents_for_cbs,
                    route_colors_global
                )
                figures_created = True
            except Exception as e:
                print(f"HATA: Grafik 1 (CBS Çakışma Çözümü) oluşturulurken: {e}")
                import traceback; traceback.print_exc()
        elif cbs_paths_found: # Sadece final çözüm varsa bile uyarı ver
             print("Grafik 1 için başlangıç çakışma verileri eksik veya dummy. Lütfen run_cbs fonksiyonunu güncelleyin. Grafik 1 atlanıyor.")

        # Grafik 2: CBS Performansının Ölçeklenmesi (Tek Çalıştırma)
        # cbs_solve_time_val ve cbs_total_cost_val yukarıda CBS çağrısı sonrası ayarlandı.
        if cbs_total_cost_val != float('inf') or cbs_solve_time_val > 0: # En az bir geçerli veri varsa
            try:
                print("Grafik 2: CBS Performansının Ölçeklenmesi (tek çalıştırma) oluşturuluyor...")
                plot_cbs_performance_scaling_single_run(num_agents, cbs_solve_time_val, cbs_total_cost_val)
                figures_created = True
            except Exception as e:
                print(f"HATA: Grafik 2 (CBS Performansı) oluşturulurken: {e}")
                import traceback; traceback.print_exc()
        elif cbs_paths_found: # Çözüm var ama maliyet/süre hesaplanamadıysa
            print("Grafik 2 için CBS çözüm süresi/maliyeti verisi eksik (veya çözüm bulunamadı), atlanıyor.")

        # Eğer animasyon gösterilmediyse ve başka grafikler oluşturulduysa, şimdi göster.
        if figures_created and (not show_animation or fig_anim is None) :
             plt.show()
        elif not figures_created:
             print("Çizilecek ek grafik bulunamadı.")

    elif not cbs_paths_found:
        print("\nCBS çözüm bulamadığı için sonuç grafikleri oluşturulmuyor.")
    elif G_osmnx_planning_graph is None: # Bu durum zaten en başta exit() ile sonlanır ama yine de kontrol.
        print("\nHarita yüklenemediği için sonuç grafikleri oluşturulmuyor.")


    # --- Bitiş ---
    end_time_main = time.time()
    print(f"\nToplam Çalışma Süresi: {end_time_main - start_time_main:.3f} saniye")
    print(f"\nBitti.")