extends Node2D

const GRID_SIZE = 10
const CELL_SIZE = 40
const PIECE_RADIUS = 18
const BOARD_MARGIN = 100

var current_player = 1  # 1为黑子，2为白子
var board = []  # 存储棋盘状态
var game_over = false
@onready var player1 = $player1/AIController2D
@onready var player2 = $player2/AIController2D

# 在文件开头添加这些常量
#const DEFAULT_PORT = 11008
#const MAX_CLIENTS = 2

# 在_ready函数开始处添加以下代码
func _ready():
	## 设置端口
	#var peer = ENetMultiplayerPeer.new()
	#peer.create_server(DEFAULT_PORT, MAX_CLIENTS)
	#multiplayer.multiplayer_peer = peer
	#print("游戏服务器运行在端口: ", DEFAULT_PORT)
	
	# 初始化棋盘数组
	for i in range(GRID_SIZE):
		board.append([])
		for j in range(GRID_SIZE):
			board[i].append(0)
	
	# 设置窗口大小
	var window_size = BOARD_MARGIN * 2 + CELL_SIZE * (GRID_SIZE - 1)
	get_window().size = Vector2(window_size, window_size)
	
	# init AI
	player1.init(self)
	player2.init(self)
	

func _draw():
	# 绘制棋盘背景
	draw_rect(Rect2(0, 0, get_window().size.x, get_window().size.y), Color(0.87, 0.72, 0.53))
	
	# 绘制网格线
	for i in range(GRID_SIZE):
		var start_x = BOARD_MARGIN + i * CELL_SIZE
		var start_y = BOARD_MARGIN + i * CELL_SIZE
		
		# 绘制垂直线
		draw_line(Vector2(start_x, BOARD_MARGIN), 
				 Vector2(start_x, BOARD_MARGIN + (GRID_SIZE-1) * CELL_SIZE),
				 Color.BLACK)
		
		# 绘制水平线
		draw_line(Vector2(BOARD_MARGIN, start_y),
				 Vector2(BOARD_MARGIN + (GRID_SIZE-1) * CELL_SIZE, start_y),
				 Color.BLACK)
	
	# 绘制棋子
	for i in range(GRID_SIZE):
		for j in range(GRID_SIZE):
			if board[i][j] != 0:
				var pos = Vector2(BOARD_MARGIN + i * CELL_SIZE,
								BOARD_MARGIN + j * CELL_SIZE)
				var color = Color.BLACK if board[i][j] == 1 else Color.WHITE
				draw_circle(pos, PIECE_RADIUS, color)
				if board[i][j] == 2:  # 为白子添加黑色轮廓
					draw_circle(pos, PIECE_RADIUS, Color.BLACK, false)

func game_over_and_reset() -> void:
	player1.done = true
	player1.needs_reset = true
	player2.done = true
	player2.needs_reset = true
	reset_game()
	

func _physics_process(_delta: float) -> void:
	if game_over:
		game_over_and_reset()
		
	if player1.needs_reset || player2.needs_reset:
		player1.reset()
		player2.reset()
		return
	
	if current_player == player1.player_id:
		if player1.control_mode == AIController2D.ControlModes.HUMAN:
			handle_human()
		else:
			handle_ai(player1)
		if player2.control_mode != AIController2D.ControlModes.HUMAN:
			handle_skip_action(player2)
	else:
		# it's player2's turn
		if player2.control_mode == AIController2D.ControlModes.HUMAN:
			handle_human()
		else:
			handle_ai(player2)
		if player1.control_mode != AIController2D.ControlModes.HUMAN:
			handle_skip_action(player1)
	return
	
func handle_ai(player : Node2D) -> void:
	if player.skip_action == true:
		player.reward -= 1000
		game_over = true
	else:
		var grid_pos = player.put_stone_action
		if is_valid_move(grid_pos):
			player.reward += check_score(grid_pos)
			place_piece(grid_pos)
			check_game_state(grid_pos)
			queue_redraw()
			update_current_player()
		else:
			player.reward -= 1000
			game_over = true

func handle_skip_action(player : Node2D) -> void:
	if player.skip_action == true:
		player.reward += 10
	else:
		player.reward -= 10

func handle_human() -> void:
	#var x = randi_range(0, 9)
	#var y = randi_range(0, 9)
	#var grid_pos = Vector2i(x, y)
	#if is_valid_move(grid_pos):
		#place_piece(grid_pos)
		#check_game_state(grid_pos)
		#queue_redraw()
		#update_current_player()

	if Input.is_action_just_pressed("mouse_click_left"):
		var mouse_pos = get_viewport().get_mouse_position()
		var grid_pos = get_grid_position(mouse_pos)
		#print("grid pos: ", grid_pos)
		if is_valid_move(grid_pos):
			place_piece(grid_pos)
			check_game_state(grid_pos)
			queue_redraw()
			update_current_player()

func update_current_player():
	#print("updating current player: original player: ", current_player)
	current_player = 1 if current_player == 2 else 2

func get_current_player():
	return current_player

func get_grid_position(mouse_pos):
	var x = round((mouse_pos.x - BOARD_MARGIN) / CELL_SIZE)
	var y = round((mouse_pos.y - BOARD_MARGIN) / CELL_SIZE)
	return Vector2i(x, y)

func is_valid_move(grid_pos):
	if grid_pos.x < 0 or grid_pos.x >= GRID_SIZE or grid_pos.y < 0 or grid_pos.y >= GRID_SIZE:
		return false
	return board[grid_pos.x][grid_pos.y] == 0

func place_piece(grid_pos):
	board[grid_pos.x][grid_pos.y] = current_player
	#print("updated board[", grid_pos.x, "][", grid_pos.y, "] to ", current_player)

func check_game_state(last_move):
	# 检查获胜条件
	if check_win(last_move):
		game_over = true
		var winner = "黑方" if current_player == 1 else "白方"
		if current_player == 1:
			player1.reward += 1e7
			player2.reward -= 1e7
		else:
			player2.reward += 1e7
			player1.reward -= 1e7
		show_game_over_dialog(winner + "获胜！")
	# 检查平局
	elif is_board_full():
		game_over = true
		show_game_over_dialog("平局！")

func check_win(pos):
	return check_score(pos) >= 5

func check_score(pos):
	var directions = [
		Vector2i(1, 0),   # 水平
		Vector2i(0, 1),   # 垂直
		Vector2i(1, 1),   # 对角线
		Vector2i(1, -1)   # 反对角线
	]
	
	var max_count = 0
	for dir in directions:
		var count = 1
		count += count_direction(pos, dir)
		count += count_direction(pos, Vector2i(-dir.x, -dir.y))
		max_count = max(max_count, count)
		if max_count >= 5:
			return 5
	return max_count

func count_direction(pos, dir):
	var count = 0
	var current = pos + dir
	
	while is_valid_position(current) and board[current.x][current.y] == current_player:
		count += 1
		current += dir
	
	return count

func is_valid_position(pos):
	return pos.x >= 0 and pos.x < GRID_SIZE and pos.y >= 0 and pos.y < GRID_SIZE

func is_board_full():
	for i in range(GRID_SIZE):
		for j in range(GRID_SIZE):
			if board[i][j] == 0:
				return false
	return true
	
func get_board():
	var board_to_return = []
	for row in board:
		board_to_return.append_array(row)
	return board_to_return

func show_game_over_dialog(message):
	return
	var dialog = AcceptDialog.new()
	add_child(dialog)
	dialog.dialog_text = message
	dialog.confirmed.connect(reset_game)
	dialog.popup_centered()

func reset_game():
	game_over = false
	current_player = 1
	for i in range(GRID_SIZE):
		for j in range(GRID_SIZE):
			board[i][j] = 0
	queue_redraw() 
