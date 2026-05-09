$(document).ready(function() {
    // Initialize the game logic (chess.js)
    var game = new Chess();
    var board = null; // Will be a chessboard.js instance
    var $status = $('#status');
    
    // Variable to store a move that's waiting for promotion
    var pendingMove = null;

    // --- API Function ---
    // This function sends the current board state to your Python server
    async function getAIMove() {
        
        // Read values from dropdowns
        var selectedAI = $('#ai-selector').val();
        var selectedTime = $('#time-selector').val();
        // Get the full game history as a PGN string
        var currentGamePGN = game.pgn(); 
        
        try {
            // Send the new data in the body
            const response = await fetch('http://localhost:5000/get_move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    fen: game.fen(),
                    pgn: currentGamePGN, // Send the PGN for history
                    ai_mode: selectedAI,  // "fast" or "mcts"
                    time_limit: parseFloat(selectedTime) // "1", "3", "5", etc.
                }),
            });

            const data = await response.json();
            
            if (data.move) {
                game.move(data.move, { sloppy: true });
                board.position(game.fen());
                updateStatus();
            }
        } catch (error) {
            console.error("Error fetching AI move:", error);
            $status.text("Error communicating with AI.");
        }
    }

    // --- Chessboard.js Callbacks ---

    // =======================================================
    // --- UPDATED onDrop FUNCTION (THIS IS THE FIX) ---
    // =======================================================
    function onDrop(source, target) {
        
        // --- 1. Test the move's legality first ---
        // We use 'q' as a temporary placeholder to see if any move is legal
        var move = game.move({
            from: source,
            to: target,
            promotion: 'q' 
        });

        // --- 2. If move is illegal, snap back ---
        // This is where 'null' will be returned if the square is blocked
        if (move === null) {
            return 'snapback';
        }

        // --- 3. If legal, undo the test move ---
        // We don't want to permanently make the move yet,
        // we just wanted to see if it was legal.
        game.undo();

        // --- 4. Now, check if it was a promotion ---
        // (We already know it's a legal move at this point)
        var piece = game.get(source);
        var isPromotion = (piece.type === 'p') &&
                          ((piece.color === 'w' && target.charAt(1) === '8') ||
                           (piece.color === 'b' && target.charAt(1) === '1'));

        if (isPromotion) {
            // --- 5. It's a LEGAL promotion, show the popup ---
            pendingMove = { from: source, to: target };
            $('#promotion-popup').show();
            // We still return 'snapback' to wait for the user's choice
            return 'snapback'; 
        }

        // --- 6. It's a LEGAL, NON-promotion move ---
        // Make the move for real this time
        game.move({
            from: source,
            to: target
        });

        // Valid move, update status and...
        updateStatus();
        
        // ...immediately call the AI to get its response
        window.setTimeout(getAIMove, 250);
    }
    // =======================================================
    // --- END OF UPDATED FUNCTION ---
    // =======================================================


    // Called after a move is made, to update the board's pieces
    function onSnapEnd() {
        board.position(game.fen());
    }

    // --- Helper Function ---
    function updateStatus() {
        var status = '';
        var moveColor = (game.turn() === 'b') ? 'Black' : 'White';

        if (game.in_checkmate()) {
            status = 'Game over, ' + moveColor + ' is in checkmate.';
        } else if (game.in_draw()) {
            status = 'Game over, drawn position';
        } else {
            status = moveColor + ' to move';
            if (game.in_check()) {
                status += ', ' + moveColor + ' is in check';
            }
        }
        $status.text(status);
    }

    // --- Click handler for the promotion popup ---
    $('#promotion-popup .promotion-piece').on('click', function() {
        var piece = $(this).data('piece');
        
        // 1. Make the stored move with the chosen piece
        game.move({
            from: pendingMove.from,
            to: pendingMove.to,
            promotion: piece
        });

        // 2. Update board, hide popup, reset pending move
        board.position(game.fen());
        $('#promotion-popup').hide();
        pendingMove = null;
        
        // 3. Update status and call the AI
        updateStatus();
        window.setTimeout(getAIMove, 250);
    });

    // --- Initialize the Board ---
    var config = {
        draggable: true,
        position: 'start',
        onDrop: onDrop,
        onSnapEnd: onSnapEnd,
        pieceTheme: 'img/chesspieces/alpha/{piece}.png' // Your local path
    };
    board = Chessboard('myBoard', config); // This line draws the board

    updateStatus();
});