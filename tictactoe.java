import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Node;
import javafx.scene.Scene;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.util.ArrayList;
import java.util.HashSet;

public class tictactoe extends Application {

    final int width = 800;
    final int heigth = 600;
    final int columns = 3;
    final int rows = 3;

    GridPane grid = new GridPane();
    Scene scene = new Scene(grid,width,heigth);

    // tracks player's cells, 0 = not marked, 1 = player1, 2 = player2
    ArrayList<Integer> field = new ArrayList<>();

    String colorBlue = "-fx-background-color:rgb(155,142,254);";
    String colorRed = "-fx-background-color:rgb(224,99,90);";


    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage stage) throws Exception {

        stage.setTitle("TicTacToe - NumPad Gameplay");

        grid.setGridLinesVisible(true);
        grid.setPadding(new Insets(20,20,20,20));

        // shape of grid
        for (int i = 0; i < columns; i++) {
            ColumnConstraints colConst = new ColumnConstraints();
            colConst.setPercentWidth(100.0 / columns);
            grid.getColumnConstraints().add(colConst);
        }
        for (int i = 0; i < rows; i++) {
            RowConstraints rowConst = new RowConstraints();
            rowConst.setPercentHeight(100.0 / rows);
            grid.getRowConstraints().add(rowConst);
        }

        // creates cells in grid, ordered as numpad
        int cellNum = 7;
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                StackPane cell = new StackPane();
                cell.setBorder(new Border(new BorderStroke(Color.BLACK,
                        BorderStrokeStyle.DOTTED, CornerRadii.EMPTY, BorderWidths.DEFAULT)));
                Label number = new Label(String.valueOf(cellNum++));
                number.setScaleX(5);
                number.setScaleY(5);
                cell.getChildren().add(number);
                grid.add(cell, j, i);
                field.add(0);
            }
            cellNum -= 6;
        }

        // tracks marked cells
        HashSet<KeyCode> blockedCells = new HashSet<>();

        // counts number of rounds, variables are not updated by keyEvents
        int[] roundsPlayed = {0};

        // applies numpad input to grid
        scene.setOnKeyPressed(e -> {
            Node node = null;
            boolean[] validInput = {true};
            String color;
            int index = -1;
            if(blockedCells.contains(e.getCode())) {
                validInput[0] = false;
            }
            if(validInput[0]){
                switch (e.getCode()){
                    case NUMPAD1 -> {
                        node = grid.getChildren().get(7);
                    }
                    case NUMPAD2 -> {
                        node = grid.getChildren().get(8);
                    }
                    case NUMPAD3 -> {
                        node = grid.getChildren().get(9);
                    }
                    case NUMPAD4 -> {
                        node = grid.getChildren().get(4);
                    }
                    case NUMPAD5 -> {
                        node = grid.getChildren().get(5);
                    }
                    case NUMPAD6 -> {
                        node = grid.getChildren().get(6);
                    }
                    case NUMPAD7 -> {
                        node = grid.getChildren().get(1);
                    }
                    case NUMPAD8 -> {
                        node = grid.getChildren().get(2);
                    }
                    case NUMPAD9 -> {
                        node = grid.getChildren().get(3);
                    }
                    case ESCAPE -> {
                        stage.close();
                        return;
                    }
                }
                blockedCells.add(e.getCode());

                // updates color & player's cells
                try{
                    index = Integer.parseInt(e.getText());
                    if(roundsPlayed[0] % 2 == 0){
                        color = colorBlue;
                        field.set(index-1,1);
                    }
                    else{
                        color = colorRed;
                        field.set(index-1,2);
                    }
                    if(node != null) node.setStyle(color);
                    roundsPlayed[0]++;
                    update(stage);
                }
                catch (NumberFormatException nfe){
                    System.out.println("invalid input");
                }
            }
        });
        stage.setResizable(false);
        stage.setScene(scene);
        stage.show();
    }

    // checks for winning lines
    public void update(Stage stage){

        int ind1, ind2, ind3;
        String winner = "";

        // horizontal
        for (int i = 0; i < 3; i++) {
            ind1 = 3*i;
            ind2 = 3*i + 1;
            ind3 = 3*i + 2;
            int firstCell = field.get(ind1), secondCell = field.get(ind2), thirdCell = field.get(ind3);
            if((firstCell == secondCell) && (firstCell == thirdCell)){
                if(firstCell == 0) continue;
                winner = firstCell == 1 ? "Blue" : "Red";
                promptWinner(stage,winner);
            }
        }

        // vertical
        for (int i = 0; i < 3; i++) {
            ind1 = i;
            ind2 = i+3;
            ind3 = i+6;
            int firstCell = field.get(ind1), secondCell = field.get(ind2), thirdCell = field.get(ind3);
            if((firstCell == secondCell) && (firstCell == thirdCell)){
                if(firstCell == 0) continue;
                winner = firstCell == 1 ? "Blue" : "Red";
                promptWinner(stage,winner);
            }
        }

        // diagonal
        for (int i = 0; i < 2; i++) {
            ind1 = 2*i;
            ind2 = 4;
            ind3 = 8-2*i;
            int firstCell = field.get(ind1), secondCell = field.get(ind2), thirdCell = field.get(ind3);
            if((firstCell == secondCell) && (firstCell == thirdCell)){
                if(firstCell == 0) continue;
                winner = firstCell == 1 ? "Blue" : "Red";
                promptWinner(stage,winner);
            }
        }

    }

    // prompts winning color
    public void promptWinner(Stage stage, String winnerColor){
        TextField text = new TextField();
        String color = winnerColor.equals("Blue") ? colorBlue : colorRed;
        text.setStyle(color);
        text.setPromptText("Game Over! - " + winnerColor + " wins!");
        text.setFocusTraversable(false);
        text.setAlignment(Pos.CENTER);
        text.setScaleX(5);
        text.setScaleY(5);
        scene.setRoot(text);
        stage.setScene(scene);
        stage.show();
    }
}
