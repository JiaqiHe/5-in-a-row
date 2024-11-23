extends AIController2D
class_name AIPlayer

var player_id : int

var put_stone_action : Vector2i = Vector2i(-1, -1)
var skip_action: bool = false

@onready var board_node = $"../.."

func reset():
	super.reset()
	skip_action = false

func set_player_id(id) -> void:
	put_stone_action = Vector2i(-1, -1)
	player_id = id

func get_obs() -> Dictionary:
	"""observation consists of:
		* 2 entries in [0 .. 1]: represents who is the current player
		* 2 entries in [2 .. 3]: represents player identity
		* 100 entries in [4 .. 103]: represents the player1's stones in the board
		* 100 entries in [104 .. 203]: represents the player2's stones in the board
	"""
	var board = board_node.get_board()
	var current_player = board_node.get_current_player()
	#var key = "obs_p" + str(self.player_id)
	var key = "obs"
	var val = []
	val.append_array([1, 0] if current_player == 1 else [0, 1]) # [0 .. 1]
	val.append_array([1, 0] if self.player_id == 1 else [0, 1]) # [2 .. 3]
	
	var stones_info_p1 = []
	var stones_info_p2 = []
	for i in board:
		stones_info_p1.push_back(1 if i == 1 else 0)
		stones_info_p2.push_back(1 if i == 2 else 0)
	val.append_array(stones_info_p1) # [4 .. 103]
	val.append_array(stones_info_p2) # [104 .. 203]
	
	return {key: val}
	
#func get_obs_space():
	## may need overriding if the obs space is complex
	#var obs = get_obs()
	#var key = "obs_p" + str(self.player_id)
	#return {
		#key: {"size": [len(obs[key])], "space": "box"},
	#}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {
		"move_action" : {
			"size": 101,
			"action_type": "discrete"
		},
		}

func set_action(action) -> void:
	if int(action["move_action"]) == 100:
		skip_action = true
		return
	skip_action = false
	var x = int(action["move_action"]) / 10
	var y = int(action["move_action"]) % 10
	put_stone_action = Vector2i(int(x), int(y))
	
