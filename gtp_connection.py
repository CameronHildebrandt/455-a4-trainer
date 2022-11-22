"""
gtp_connection.py
Module for playing games of Go using GoTextProtocol

Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller.
Parts of this code were originally based on the gtp module
in the Deep-Go project by Isaac Henrion and Amos Storkey
at the University of Edinburgh.
"""
import traceback
import numpy as np
import re
from sys import stdin, stdout, stderr
from typing import Any, Callable, Dict, List, Tuple
import time # imported by us
import random # imported by us
import os # imported by us

from board_base import (
    is_black_white,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    MAXSIZE,
    coord_to_point,
    opponent
)
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine


class TranspositionTable(object):
# Table is stored in a dictionary, with board code as key, 
# and minimax score as the value

    # Empty dictionary
    def __init__(self):
        self.table = {}
        
    # Used to print the whole table with print(tt)
    def __repr__(self):
        return self.table.__repr__()
        
    def store(self, code, score):
        self.table[code] = score
    
    # Python dictionary returns 'None' if key not found by get()
    def lookup(self, code):
        # print("simplification for", code, " - ", self.table.get(code))
        return self.table.get(code)


class GtpConnection:
    def __init__(self, go_engine: GoEngine, board: GoBoard, debug_mode: bool = False) -> None:
        """
        Manage a GTP connection for a Go-playing engine

        Parameters
        ----------
        go_engine:
            a program that can reply to a set of GTP commandsbelow
        board:
            Represents the current board state.
        """
        self._debug_mode: bool = debug_mode
        self.go_engine = go_engine
        self.board: GoBoard = board
        self.commands: Dict[str, Callable[[List[str]], None]] = {
            "protocol_version": self.protocol_version_cmd,
            "quit": self.quit_cmd,
            "name": self.name_cmd,
            "boardsize": self.boardsize_cmd,
            "showboard": self.showboard_cmd,
            "clear_board": self.clear_board_cmd,
            "komi": self.komi_cmd,
            "version": self.version_cmd,
            "known_command": self.known_command_cmd,
            "genmove": self.genmove_cmd,
            "list_commands": self.list_commands_cmd,
            "play": self.play_cmd,
            "legal_moves": self.legal_moves_cmd,
            "gogui-rules_legal_moves": self.gogui_rules_legal_moves_cmd,
            "gogui-rules_final_result": self.gogui_rules_final_result_cmd,
            "solve": self.solve_cmd,
            "simulate": self.simulate_cmd,
            "timelimit": self.time_limit_cmd,
        }

        # argmap is used for argument checking
        # values: (required number of arguments,
        #          error message on argnum failure)
        self.argmap: Dict[str, Tuple[int, str]] = {
            "boardsize": (1, "Usage: boardsize INT"),
            "komi": (1, "Usage: komi FLOAT"),
            "known_command": (1, "Usage: known_command CMD_NAME"),
            "genmove": (1, "Usage: genmove {w,b}"),
            "play": (2, "Usage: play {b,w} MOVE"),
            "legal_moves": (1, "Usage: legal_moves {w,b}"),
        }
        self.time_limit = 1
        self.start_time = 0
        self.winning_move = None
        self.moveOrder = [-1 for _ in range(49)]
        self.depth = 0 # the amount of moves that have been played by the solver
        self.moveOrderLocked = False
        self.sampleNumber = 0
        self.tt = TranspositionTable() # use separate table for each color

    def write(self, data: str) -> None:
        stdout.write(data)

    def flush(self) -> None:
        stdout.flush()

    def start_connection(self) -> None:
        """
        Start a GTP connection.
        This function continuously monitors standard input for commands.
        """
        line = stdin.readline()
        while line:
            self.get_cmd(line)
            line = stdin.readline()

    def get_cmd(self, command: str) -> None:
        """
        Parse command string and execute it
        """
        if len(command.strip(" \r\t")) == 0:
            return
        if command[0] == "#":
            return
        # Strip leading numbers from regression tests
        if command[0].isdigit():
            command = re.sub("^\d+", "", command).lstrip()

        elements: List[str] = command.split()
        if not elements:
            return
        command_name: str = elements[0]
        args: List[str] = elements[1:]
        if self.has_arg_error(command_name, len(args)):
            return
        if command_name in self.commands:
            try:
                self.commands[command_name](args)
            except Exception as e:
                self.debug_msg("Error executing command {}\n".format(str(e)))
                self.debug_msg("Stack Trace:\n{}\n".format(traceback.format_exc()))
                raise e
        else:
            self.debug_msg("Unknown command: {}\n".format(command_name))
            self.error("Unknown command")
            stdout.flush()

    def has_arg_error(self, cmd: str, argnum: int) -> bool:
        """
        Verify the number of arguments of cmd.
        argnum is the number of parsed arguments
        """
        if cmd in self.argmap and self.argmap[cmd][0] != argnum:
            self.error(self.argmap[cmd][1])
            return True
        return False

    def debug_msg(self, msg: str) -> None:
        """ Write msg to the debug stream """
        if self._debug_mode:
            stderr.write(msg)
            stderr.flush()

    def error(self, error_msg: str) -> None:
        """ Send error msg to stdout """
        stdout.write("? {}\n\n".format(error_msg))
        stdout.flush()

    def respond(self, response: str = "") -> None:
        """ Send response to stdout """
        stdout.write("= {}\n\n".format(response))
        stdout.flush()

    def reset(self, size: int) -> None:
        """
        Reset the board to empty board of given size
        """
        self.board.reset(size)

    def board2d(self) -> str:
        return str(GoBoardUtil.get_twoD_board(self.board))

    def protocol_version_cmd(self, args: List[str]) -> None:
        """ Return the GTP protocol version being used (always 2) """
        self.respond("2")

    def quit_cmd(self, args: List[str]) -> None:
        """ Quit game and exit the GTP interface """
        self.respond()
        exit()

    def name_cmd(self, args: List[str]) -> None:
        """ Return the name of the Go engine """
        self.respond(self.go_engine.name)

    def version_cmd(self, args: List[str]) -> None:
        """ Return the version of the  Go engine """
        self.respond(str(self.go_engine.version))

    def clear_board_cmd(self, args: List[str]) -> None:
        """ clear the board """
        self.reset(self.board.size)
        self.respond()

    def boardsize_cmd(self, args: List[str]) -> None:
        """
        Reset the game with new boardsize args[0]
        """
        self.reset(int(args[0]))
        self.respond()

    def showboard_cmd(self, args: List[str]) -> None:
        self.respond("\n" + self.board2d())

    def komi_cmd(self, args: List[str]) -> None:
        """
        Set the engine's komi to args[0]
        """
        self.go_engine.komi = float(args[0])
        self.respond()

    def known_command_cmd(self, args: List[str]) -> None:
        """
        Check if command args[0] is known to the GTP interface
        """
        if args[0] in self.commands:
            self.respond("true")
        else:
            self.respond("false")

    def list_commands_cmd(self, args: List[str]) -> None:
        """ list all supported GTP commands """
        self.respond(" ".join(list(self.commands.keys())))
        
        
        
    def legal_moves_cmd(self, args: List[str]) -> None:
        """
        List legal moves for color args[0] in {'b','w'}
        """
        board_color: str = args[0].lower()
        color: GO_COLOR = color_to_int(board_color)
        moves: List[GO_POINT] = GoBoardUtil.generate_legal_moves(self.board, color)
        gtp_moves: List[str] = []
        for move in moves:
            coords: Tuple[int, int] = point_to_coord(move, self.board.size)
            gtp_moves.append(format_point(coords))
        sorted_moves = " ".join(sorted(gtp_moves))
        self.respond(sorted_moves)
        
        
        
    """
    ==========================================================================
    Assignment 2 - game-specific commands start here
    ==========================================================================
    """
    """
    ==========================================================================
    Assignment 2 - commands we already implemented for you
    ==========================================================================
    """
        
        

    def gogui_analyze_cmd(self, args):
        """ We already implemented this function for Assignment 2 """
        self.respond("pstring/Legal Moves For ToPlay/gogui-rules_legal_moves\n"
                     "pstring/Side to Play/gogui-rules_side_to_move\n"
                     "pstring/Final Result/gogui-rules_final_result\n"
                     "pstring/Board Size/gogui-rules_board_size\n"
                     "pstring/Rules GameID/gogui-rules_game_id\n"
                     "pstring/Show Board/gogui-rules_board\n"
                     )

    def gogui_rules_game_id_cmd(self, args):
        """ We already implemented this function for Assignment 2 """
        self.respond("NoGo")

    def gogui_rules_board_size_cmd(self, args):
        """ We already implemented this function for Assignment 2 """
        self.respond(str(self.board.size))

    def gogui_rules_side_to_move_cmd(self, args):
        """ We already implemented this function for Assignment 2 """
        color = "black" if self.board.current_player == BLACK else "white"
        self.respond(color)

    def gogui_rules_board_cmd(self, args):
        """ We already implemented this function for Assignment 2 """
        size = self.board.size
        str = ''
        for row in range(size-1, -1, -1):
            start = self.board.row_start(row + 1)
            for i in range(size):
                #str += '.'
                point = self.board.board[start + i]
                if point == BLACK:
                    str += 'X'
                elif point == WHITE:
                    str += 'O'
                elif point == EMPTY:
                    str += '.'
                else:
                    assert False
            str += '\n'
        self.respond(str)
        
        
    
    def gogui_rules_legal_moves_cmd(self, args):
        # get all the legal moves
        legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
        coords = [point_to_coord(move, self.board.size) for move in legal_moves]
        # convert to point strings
        point_strs  = [ chr(ord('a') + col - 1) + str(row) for row, col in coords]
        point_strs.sort()
        point_strs = ' '.join(point_strs).upper()
        self.respond(point_strs)
        return
        


    """
    ==========================================================================
    Assignment 2 - game-specific commands you have to implement or modify
    ==========================================================================
    """
    def gogui_rules_final_result_cmd(self, args):
        """ Implement this method correctly """
        
        legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
        if len(legal_moves) > 0:
            self.respond('unknown')
        elif self.board.current_player == BLACK:
            self.respond('w')
        else:
            self.respond('b')

    def play_cmd(self, args: List[str]) -> None:
        """
        play a move args[1] for given color args[0] in {'b','w'}
        """
        # change this method to use your solver
        try:
            board_color = args[0].lower()
            board_move = args[1]
            color = color_to_int(board_color)
            
            coord = move_to_coord(args[1], self.board.size)
            if coord:
                move = coord_to_point(coord[0], coord[1], self.board.size)
            else:
                self.error(
                    "Error executing move {} converted from {}".format(move, args[1])
                )
                return
            
            success = self.board.play_move(move, color)
            if not success:
                self.respond('illegal move')
                return
            else:
                self.debug_msg(
                    "Move: {}\nBoard:\n{}\n".format(board_move, self.board2d())
                )
            self.respond()
        except Exception as e:
            self.respond("Error: {}".format(str(e)))
            

    def _genmove(self, state):
        """ same as the genmove_cmd, but returns instead of responds so it can be used internally """

        win, move = None, None
        isRandomPlay = False

        # try:
        win, move = self._solve()
        # except: # commented out so that we can manually stop training
            # print("some error happened in solve")
            # return False, isRandomPlay

        if move is None:
            # print("move is none")
            return False, isRandomPlay

        if(not win):
            isRandomPlay = True
            legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
            move = random.choice(legal_moves)


            
        move_coord = point_to_coord(move, self.board.size)

        if self.board.is_legal(move, state.current_player):
            self.board.play_move(move, state.current_player)
            return move_coord, isRandomPlay
        else:
            print("====================================================================")
            print("ERROR - genmove attempted an illegal move - something is broken")
            legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
            print("Legal moves: ", moves_readable(legal_moves, self.board.size))
            print("Move attempted: ", moves_readable([move], self.board.size))
            print("====================================================================")
            return False, isRandomPlay
    

    def genmove_cmd(self, args: List[str]) -> None:
        """ generate a move for color args[0] in {'b','w'} """
        board_color = args[0].lower()
        color = color_to_int(board_color)

        win, move = None, None

        try:
            win, move = self._solve()
        except:
            self.respond('unknown') # Time limit exceeded
            return

        if(not win):
            self.respond('resign')
            return

        if move is None:
            self.respond('unknown')
            return

            
        move_coord = point_to_coord(move, self.board.size)
        move_as_string = format_point(move_coord)

        if self.board.is_legal(move, color):
            self.board.play_move(move, color)
            self.respond(move_as_string)
        else:
            self.respond("Illegal move: {}".format(move_as_string))

    def minimaxBooleanOR(self, state):
        # result = self.tt.lookup(state.codeSpecialHash()) # TT
        # if result != None: # TT
        #     return result # TT


        # use an array that keeps getting passed down - maybe save to global state and lock from changes once we see a win?

        # Time limit check
        # timeUsed = time.time() - self.start_time
        # if timeUsed > self.time_limit:
        #     raise Exception("Time limit exceeded")

        # ===============================================

        # if(self.winning_move != None):
        #     move_to_encode = [self.winning_move]

        #     coords = [point_to_coord(move, self.board.size) for move in move_to_encode]
        #     # convert to point strings
        #     point_strs  = [ chr(ord('a') + col - 1) + str(row) for row, col in coords]
        #     point_strs.sort()
        #     point_strs = ' '.join(point_strs).upper()

            # print("WM: ", point_strs)

        # ===============================================


        legal_moves = GoBoardUtil.generate_legal_moves(state, state.current_player)
        end_of_game = len(legal_moves) == 0

        if end_of_game:
            # self.tt.store(state.codeSpecialHash(), False) # TT
            return False
        
        for m in legal_moves:
            state.play_move(m, state.current_player)

            if(not self.moveOrderLocked):
                self.moveOrder[self.depth] = m

            # result = self.tt.lookup(state.codeSpecialHash()) # TT
            # if result != None: # TT
                # print("USING STORED VALUE | Code: ", state.codeSpecialHash(), " | res: ", result)
            #     isWin = result # TT
            # else:

            self.depth += 1
            isWin = self.minimaxBooleanAND(state)
            self.depth -= 1

            # self.tt.store(state.codeSpecialHash(), isWin) # TT
            
            # if result != None:
            #     if result != isWin:
            #         print('BAD')
            # if state.codeSpecialHash() == '1112221022211102':
            #     print(self.tt.lookup(state.codeSpecialHash()))

            state.undo_move(m)
            if isWin:
                # save the move that made win - play that
                self.winning_move = m
                self.moveOrderLocked = True

                # if result != None:
                #     if result != True: print("result wrong 1!")
                #     return result
                
                # state.play_move(m, state.current_player)
                # self.tt.store(state.codeSpecialHash(), True) # TT
                # state.undo_move(m)
                return True

        # if result != None:
        #     if result != False: print("result wrong 2!")
        #     return result

        # self.tt.store(state.codeSpecialHash(), False) # TT
        return False

    def minimaxBooleanAND(self, state):
        # result = self.tt.lookup(state.codeSpecialHash()) # TT
        # if result != None: # TT
        #     return result # TT

        legal_moves = GoBoardUtil.generate_legal_moves(state, state.current_player)
        end_of_game = len(legal_moves) == 0

        if end_of_game:
            # self.tt.store(state.codeSpecialHash(), True) # TT
            return True

        for m in legal_moves:
            state.play_move(m, state.current_player)

            # if(not self.moveOrderLocked):
            #     self.moveOrder[self.depth] = m
            
            # result = self.tt.lookup(state.codeSpecialHash()) # TT
            # if result != None: # TT
            #     isLoss = result # TT
            # else:

            # self.depth += 1
            isLoss = not self.minimaxBooleanOR(state)
            # self.depth -= 1

            # self.tt.store(state.codeSpecialHash(), isLoss) # TT

            state.undo_move(m)
            if isLoss:
                # if result != None:
                #     if result != False: print("result wrong 3!")
                    # return result

                # state.play_move(m, state.current_player)
                # self.tt.store(state.codeSpecialHash(), False) # TT
                # state.undo_move(m)
                return False

        # if result != None:
        #     if result != True: print("result wrong 4!")
            # return result

        # self.tt.store(state.codeSpecialHash(), True) # TT
        # self.moveOrderLocked = True
        return True


    def _solve(self) -> str:
        self.start_time = time.time()

        state = self.board.copy()
        
        win = False
        try:
            win = self.minimaxBooleanOR(state)
        except:
            raise
        finally:
            print("time: ", time.time() - self.start_time)

        return (win, self.winning_move)
            
            
    def solve_cmd(self, args: List[str]) -> None:
        win, move = None, None

        try:
            win, move = self._solve()
        except:
            self.respond('unknown') # Time limit exceeded
            return

        # print("tt:", self.tt)

        if(win):
            coord: Tuple[int, int] = point_to_coord(move, self.board.size)
            winning_move = str(format_point(coord)).lower()

            if(self.board.current_player == BLACK):
                self.respond('b ' + winning_move)
            else:
                self.respond('w ' + winning_move)

        else:
            if(self.board.current_player == BLACK):
                self.respond('w')
            else:
                self.respond('b')




    def simulate_cmd(self, args: List[str]) -> None:
        # this command will
        # 1: play n random moves
        # 2: loop until game ends
        # 2a: genmove from that state
        # 2b: save the chosen move and state


        # Read index file (if exists, else create it)
        # from index - save the current training sample number to the global scope
        if(os.path.exists("index.npy")):
            self.sampleNumber = np.load("index.npy") # should only be a single number - no error handling for corrupt index file
        else:
            np.save("index.npy", [0])


        ### The main training loop - run until stopped
        while(True):
            # Reset the board for another run
            self.reset(self.board.size)

            state = self.board

            handoffThreshold = 35 # this is n
            # handoffThreshold = 38 # this is n
            moveNumber = 0


            # 1: play n random moves
            while(moveNumber < handoffThreshold):
                # list all legal moves
                legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)


                # pick a random one (TODO: add heuristics)
                # easy heuristics to add:
                # - distance from other pieces
                # - distance from the wall (ideally ~2)

                m = random.choice(legal_moves)
                state.play_move(m, state.current_player)

                # no need to swap player, this is handled in play_move
                moveNumber += 1


            # 2: use the solver to play out the rest of the game
            legal_moves = GoBoardUtil.generate_legal_moves(state, state.current_player)
            while(len(legal_moves) != 0):

                # save the board state before generating move
                board = GoBoardUtil.get_twoD_board(self.board)
                oneHotBoard = board_to_one_hot_encoding(board, state.current_player)

                print("================================================================")
                print("Generating move...")
                print("Legal moves: ", moves_readable(legal_moves ,7))
                print("Current Player: ", state.current_player)
                print("Board:\n", self.board2d())
                print("================================================================")


                # genmove changes the state, so no need to play move here
                coord, isRandomPlay = self._genmove(state)

                if(coord == False):
                    # something unexpected happened - we can't use this as a training example, so skip
                    # you shouldn't see this in training
                    print("SKIPPING A TRAINING EXAMPLE DUE TO ERROR")
                else:
                    if(not isRandomPlay):
                        print("Saving best move", coords_readable([coord]), "as sample number", self.sampleNumber)

                        # Save board position (train)
                        np.save("train/" + str(self.sampleNumber) + ".npy", oneHotBoard)

                        # Save move (label)
                        np.save("label/" + str(self.sampleNumber) + ".npy", coord_to_one_hot_vector(coord, 7))

                        # Update the sampleNumber + index
                        self.sampleNumber += 1
                        np.save("index.npy", self.sampleNumber)

                    else:
                        print("The current player is losing so they played random - not saving the move")


                # make sure to update legal moves at end, or else infinite loop
                legal_moves = GoBoardUtil.generate_legal_moves(state, state.current_player)










    ##### First attempt at simulate (doesn't work) vvvvvvvv
    # def simulate_cmd(self, args: List[str]) -> None:
    #     # this command will
    #     # 1: play n random moves
    #     # 2: run solve from that state
    #     # 3: save the game boards and chosen moves to the dataset

    #     # handoffThreshold = 34 # this is n
    #     handoffThreshold = 38 # this is n
    #     moveNumber = 0

    #     # state = self.board.copy()
    #     state = self.board


    #     # 1: play n random moves
    #     while(moveNumber < handoffThreshold):
    #         # list all legal moves
    #         legal_moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)


    #         # pick a random one (TODO: add heuristics)
    #         m = random.choice(legal_moves)
    #         state.play_move(m, state.current_player)

    #         # no need to swap player, this is handled in play_move
    #         moveNumber += 1


    #     # 2: run solve from that state
    #     self._solve()

    #     # ===============================================

    #     if(self.winning_move != None):
    #         coords = []
    #         for move in self.moveOrder:
    #             if(move != -1):
    #                 coords.append(point_to_coord(move, self.board.size))

    #         point_strs  = [ chr(ord('a') + col - 1) + str(row) for row, col in coords]
    #         point_strs.sort()
    #         point_strs = ' '.join(point_strs).upper()

    #         print("moveOrder: ", point_strs)

    #     # ===============================================


    #     # 3: save the gameboards and chosen moves to the dataset

    #     # don't bother saving the moves when it leads to a loss

    #     # save the random moves in a different place than the generated moves

    #     # may be able to double data if we can record opponent moves too (is it valid to assume that the and node was the best move tho?)

    #     ##### First attempt at simulate (doesn't work) ^^^^^^^





    def time_limit_cmd(self, args: List[str]):
        upper = 100
        lower = 1
        
        seconds = int(args[0])

        if(seconds > upper or seconds < lower):
            self.respond(str(seconds) + ' is out of range [' + str(lower) + ',' + str(upper) + ']')
            return
        else:
            self.time_limit = seconds
            self.respond('time limit set to ' + str(self.time_limit))

    """
    ==========================================================================
    Assignment 2 - game-specific commands end here
    ==========================================================================
    """

def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index
    to (row, col) coordinate representation.
    """
    NS = boardsize + 1
    return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1'
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    row, col = move
    return column_letters[col - 1] + str(row)


def move_to_coord(point_str: str, board_size: int) -> Tuple[int, int]:
    """
    Convert a string point_str representing a point, as specified by GTP,
    to a pair of coordinates (row, col) in range 1 .. board_size.
    
    """
    s = point_str.lower()
    col_c = s[0]
    col = ord(col_c) - ord("a")
    if col_c < "i":
        col += 1
    row = int(s[1:])
        
    return row, col



def color_to_int(c: str) -> int:
    """convert character to the appropriate integer code"""
    color_to_int = {"b": BLACK, "w": WHITE, "e": EMPTY, "BORDER": BORDER}
    return color_to_int[c]


def coord_to_one_hot_vector(coord, boardsize):
    boardsize = 7 # hardcoded to prevent silly mistakes

    # Coords are 1-based, we need 0 based for the conversion
    x = coord[0] - 1
    y = coord[1] - 1

    numberRep = x + y * boardsize
    arrayRep = [0 for _ in range(boardsize * boardsize)]
    arrayRep[numberRep] = 1

    return arrayRep

def board_to_one_hot_encoding(board, current_player):
    current_players_stones = [[0 for _ in range(len(board))] for _ in range(len(board))]
    opponent_stones = [[0 for _ in range(len(board))] for _ in range(len(board))]

    for row in range(len(board)):
        for col in range(len(board)):
            if(board[row][col] == 0): # empty space, skip
                continue
            if(board[row][col] == current_player):
                current_players_stones[row][col] = 1
            else:
                opponent_stones[row][col] = 1

    return np.array([current_players_stones, opponent_stones])



def moves_readable(moves, boardsize):
    coords = [point_to_coord(moves, boardsize) for moves in moves]
    point_strs  = [ chr(ord('a') + col - 1) + str(row) for row, col in coords]
    point_strs.sort()
    return ' '.join(point_strs).upper()

def coords_readable(coords):
    point_strs  = [ chr(ord('a') + col - 1) + str(row) for row, col in coords]
    point_strs.sort()
    return ' '.join(point_strs).upper()