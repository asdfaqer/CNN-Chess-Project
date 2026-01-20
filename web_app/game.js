var board = null;
var game = new Chess();
var $status = $('#status');
var $pgn = $('.pgn-content');
var evalChart = null;
var evalHistory = [0];

var API_URL = "http://localhost:5000";

function onDragStart (source, piece, position, orientation) {
  // Do not pick up pieces if the game is over
  if (game.game_over()) return false;

  // Only pick up pieces for White (Human) or if it's currently that color's turn
  // In this app, we assume human is White for simplicity
  if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
      (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    return false;
  }
}

function makeBotMove() {
    console.log("Requesting bot move...");
    $.ajax({
        url: API_URL + "/get_move",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            fen: game.fen(),
            pgn: game.pgn()
        }),
        success: function(data) {
            console.log("Bot move received:", data);
            if (data.move) {
                game.move(data.move, { sloppy: true });
                board.position(game.fen());
                updateStatus();
                
                if (data.top_moves) {
                    updateTopMoves(data.top_moves);
                    // Add the best score to history for chart
                    const bestScore = data.top_moves[0].score;
                    updateEvalChart(bestScore);
                }
            }
        },
        error: function(err) {
            console.error("Error getting bot move:", err);
            $('#model-status-text').text("Error").parent().find('.status-indicator').addClass('error').removeClass('online');
        }
    });
}

function onDrop (source, target) {
  // See if the move is legal
  var move = game.move({
    from: source,
    to: target,
    promotion: 'q' // NOTE: always promote to a queen for example simplicity
  });

  // Illegal move
  if (move === null) return 'snapback';

  updateStatus();
  
  // Wait for board to update before bot moves
  window.setTimeout(makeBotMove, 250);
}

// Update the board position after the piece snap
// for castling, en passant, pawn promotion
function onSnapEnd () {
  board.position(game.fen());
}

function updateStatus () {
  var status = '';

  var moveColor = 'White';
  if (game.turn() === 'b') {
    moveColor = 'Black';
  }

  // Checkmate?
  if (game.in_checkmate()) {
    status = 'Game over, ' + moveColor + ' is in checkmate.';
  }
  // Draw?
  else if (game.in_draw()) {
    status = 'Game over, drawn position';
  }
  // Game still on
  else {
    status = moveColor + ' to move';
    // Check?
    if (game.in_check()) {
      status += ', ' + moveColor + ' is in check';
    }
  }

  $pgn.html(game.pgn());
}

function updateTopMoves(moves) {
    const list = $('#top-moves-list');
    list.empty();
    
    moves.forEach(m => {
        const item = $(`
            <div class="move-item">
                <span class="move-name">${m.move}</span>
                <span class="move-score">${m.score.toFixed(2)}</span>
            </div>
        `);
        list.append(item);
    });
}

function initEvalChart() {
    const ctx = document.getElementById('evalChart').getContext('2d');
    evalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['0'],
            datasets: [{
                label: 'Evaluation (Logits)',
                data: [0],
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    grid: { color: '#334155' },
                    ticks: { color: '#94a3b8' }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false }
                }
            }
        }
    });
}

function updateEvalChart(newScore) {
    evalHistory.push(newScore);
    evalChart.data.labels.push(evalHistory.length.toString());
    evalChart.data.datasets[0].data.push(newScore);
    evalChart.update();
}

function checkModelStatus() {
    $.get(API_URL + "/status", function(data) {
        if (data.status === "ready") {
            $('#model-status-text').text("Online (" + data.device + ")");
            $('.status-indicator').addClass('online').removeClass('loading');
        }
    }).fail(function() {
        $('#model-status-text').text("Offline");
        $('.status-indicator').addClass('error').removeClass('loading');
    });
}

// Reset Game
$('#reset-btn').on('click', function() {
    game.reset();
    board.start();
    updateStatus();
    evalHistory = [0];
    evalChart.data.labels = ['0'];
    evalChart.data.datasets[0].data = [0];
    evalChart.update();
    $('#top-moves-list').html('<div class="move-item empty">Play a move to see analysis</div>');
});

// Flip Board
$('#flip-btn').on('click', function() {
    board.flip();
});

var config = {
  draggable: true,
  position: 'start',
  onDragStart: onDragStart,
  onDrop: onDrop,
  onSnapEnd: onSnapEnd,
  pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png'
};

board = ChessBoard('myBoard', config);

updateStatus();
initEvalChart();
checkModelStatus();
setInterval(checkModelStatus, 10000);
