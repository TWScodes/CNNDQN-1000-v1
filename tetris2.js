class Tetris {
    constructor(width = 10, height = 20, score = 0, gameNumber = 0, gameOver = false) {
        // ----- Dimensions & Board ----- //
        this.width = width;
        this.height = height;
        this.board = Array.from({ length: this.height }, () => Array(this.width).fill(0)); // Create board filled with zeros

        // ----- Stats ----- //
        this.score = score;
        this.gameNumber = gameNumber;
        this.gameOver = gameOver;

        // ----- Pieces ----- //
        this.tetrominos = {
            o: [[1, 1], [1, 1]],
            i: [[1], [1], [1], [1]],
            s: [[0, 1, 1], [1, 1, 0]],
            z: [[1, 1, 0], [0, 1, 1]],
            l: [[1, 0], [1, 0], [1, 1]],
            j: [[0, 1], [0, 1], [1, 1]],
            t: [[1, 1, 1], [0, 1, 0]],
        };
    }

    log() { // Log board for debugging purposes
        console.log(this.board);
    }

    spawn(type, pos) { // Spawn a tetromino
        const shape = this.tetrominos[type];
        if (!shape) {
            console.error(`Invalid tetromino type: ${type}`);
            return;
        }

        let { x, y } = pos;
        this.activeTetromino = { shape, x, y };
        this.placeTetromino();
    }

    placeTetromino() {
        let { shape, x, y } = this.activeTetromino;
        shape.forEach((row, rowIndex) => {
            row.forEach((cell, colIndex) => {
                if (cell === 1) {
                    this.board[y + rowIndex][x + colIndex] = 1;
                }
            });
        });
    }

    rotateTetromino() {
        let { shape, x, y } = this.activeTetromino;
        const rotatedShape = shape[0].map((_, index) => shape.map(row => row[index])).reverse();

        // Check if the rotated shape fits
        if (this.canPlaceTetromino(rotatedShape, x, y)) {
            this.activeTetromino.shape = rotatedShape;
            this.clearTetromino();
            this.placeTetromino();
        }
    }

    canPlaceTetromino(shape, x, y) {
        return shape.every((row, rowIndex) => {
            return row.every((cell, colIndex) => {
                if (cell === 1) {
                    const boardX = x + colIndex;
                    const boardY = y + rowIndex;
                    return boardX >= 0 && boardX < this.width && boardY < this.height && this.board[boardY][boardX] === 0;
                }
                return true;
            });
        });
    }

    clearTetromino() {
        let { shape, x, y } = this.activeTetromino;
        shape.forEach((row, rowIndex) => {
            row.forEach((cell, colIndex) => {
                if (cell === 1) {
                    this.board[y + rowIndex][x + colIndex] = 0;
                }
            });
        });
    }

    step1() {
        // Handle line clearing
        for (let rowIndex = this.height - 1; rowIndex >= 0; rowIndex--) {
            if (this.board[rowIndex].every(cell => [1, 2].includes(cell))) {
                this.score += 5; // 5 points for line clear
                // Shift rows down
                for (let i = rowIndex; i > 0; i--) {
                    this.board[i] = [...this.board[i - 1]];
                }
                this.board[0] = Array(this.width).fill(0); // Fill the top row with '0's after shift
            }
        }
    }

    step2_3() {
        const randomTetromino = Object.keys(this.tetrominos)[Math.floor(Math.random() * Object.keys(this.tetrominos).length)];

        // Check if the board includes live cells (1)
        if (!this.board.flat().includes(1)) {
            // Check if the top middle position is empty to spawn the new piece
            if (this.board[0][Math.floor(this.width / 2)] === 0) {
                this.spawn(randomTetromino, { x: Math.floor(this.width / 2), y: 0 });
            } else {
                this.gameOver = true; // Game over if the spawn position is blocked
            }
        }
    }

    step4(action) {
        const move = (direction) => {
            this.board.forEach((row, rowIndex) => {
                row.forEach((cell, colIndex) => {
                    if (cell === 1) {
                        let newX = colIndex, newY = rowIndex;

                        if (direction === 'left' && newX > 0 && this.board[newY][newX - 1] === 0) {
                            newX--;
                        } else if (direction === 'right' && newX < this.width - 1 && this.board[newY][newX + 1] === 0) {
                            newX++;
                        } else if (direction === 'down' && newY < this.height - 1 && this.board[newY + 1][newX] === 0) {
                            newY++;
                        }

                        if (newX !== colIndex || newY !== rowIndex) {
                            this.board[rowIndex][colIndex] = 0;
                            this.board[newY][newX] = 1;
                        }
                    }
                });
            });
        };

        if (['left', 'right', 'down'].includes(action)) {
            move(action);
        } else if (action === 'rotate') {
            this.rotateTetromino();
        }
    }

    step5() {
        let canMove = true;

        // Apply gravity
        for (let rowIndex = this.height - 2; rowIndex >= 0; rowIndex--) {
            for (let colIndex = 0; colIndex < this.width; colIndex++) {
                if (this.board[rowIndex][colIndex] === 1 && this.board[rowIndex + 1][colIndex] === 0) {
                    // Move the block down
                    this.board[rowIndex + 1][colIndex] = 1;
                    this.board[rowIndex][colIndex] = 0;
                } else if (this.board[rowIndex][colIndex] === 1 && this.board[rowIndex + 1][colIndex] === 2) {
                    // Freeze the block if it can't move down
                    this.board[rowIndex][colIndex] = 2;
                }
            }
        }

        // If pieces cannot move, stop the game
        if (this.board[0].includes(1)) {
            this.gameOver = true;
        }
    }

    step(action = '') {
        // #1 Check for line clears and score
        this.step1();

        // #2 and #3 Spawn New Tetromino (if needed) & Check for Game Over
        this.step2_3();

        // #4 Handle Player Input (Move)
        this.step4(action);

        // #5 Apply Gravity (Automatic Fall)
        this.step5();
    }
}
