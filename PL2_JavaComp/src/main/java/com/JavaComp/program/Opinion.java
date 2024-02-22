package com.JavaComp.program;

import java.io.Serializable;
import java.time.LocalDate;

public class Opinion implements Serializable {
    private int calif;
    private String comentario;
    private LocalDate fecha;

    public Opinion(int calif, String comentario, LocalDate fecha) {
        this.calif = calif;
        this.comentario = comentario;
        this.fecha = fecha;
    }

    public int getCalif() {
        return calif;
    }

    public void setCalif(int calif) {
        this.calif = calif;
    }

    public String getComentario() {
        return comentario;
    }

    public void setComentario(String comentario) {
        this.comentario = comentario;
    }

    public LocalDate getFecha() {
        return fecha;
    }

    public void setFecha(LocalDate fecha) {
        this.fecha = fecha;
    }
}
