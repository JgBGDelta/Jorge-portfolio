/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package com.JavaComp.interf;

import com.JavaComp.program.*;
import java.awt.Image;
import java.util.ArrayList;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;
/**
 *
 * @author Kar�n
 */
public class InterfCliente extends javax.swing.JFrame {

    /**
     * Crea un InterfCliente y le asigna todos los atributos además de cargar la
     * lista de productos ordenada por título y por relevancia.
     */
    public InterfCliente() {
        initComponents();
        ArrayList<Producto> array = (ArrayList) DataManager.getProductos().clone();
        
        ImageIcon imageIcon = new ImageIcon("src/main/resources/images/LogoJavaComp.png"); // load the image to a imageIcon
        Image image = imageIcon.getImage(); // transform it 
        Image newimg = image.getScaledInstance(45, 45,  java.awt.Image.SCALE_SMOOTH); // scale it the smooth way  
        imageIcon = new ImageIcon(newimg);
        iconImageLabel.setIcon(imageIcon);
        
        busquedaScroll.getVerticalScrollBar().setUnitIncrement(16);
        
        DataManager.sortTitulo(array);
        DataManager.sortRelevancia(array);
        DataManager.displayList(array, busquedaPanel);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        utilCalendarModel1 = new org.jdatepicker.UtilCalendarModel();
        jPasswordField1 = new javax.swing.JPasswordField();
        jCheckBox1 = new javax.swing.JCheckBox();
        editarPerfilBoton = new javax.swing.JButton();
        categoriaBox = new javax.swing.JComboBox<>();
        categoria = new javax.swing.JLabel();
        ordenarPor = new javax.swing.JLabel();
        ordenarBox = new javax.swing.JComboBox<>();
        busquedaScroll = new javax.swing.JScrollPane();
        busquedaPanel = new javax.swing.JPanel();
        jCheckBox2 = new javax.swing.JCheckBox();
        jCheckBox3 = new javax.swing.JCheckBox();
        jCheckBox4 = new javax.swing.JCheckBox();
        jCheckBox5 = new javax.swing.JCheckBox();
        jCheckBox6 = new javax.swing.JCheckBox();
        jCheckBox7 = new javax.swing.JCheckBox();
        carroBoton = new javax.swing.JButton();
        buscarBoton = new javax.swing.JButton();
        buscarField = new javax.swing.JTextField();
        cerrarSesionBoton = new javax.swing.JButton();
        iconImageLabel = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();

        jPasswordField1.setText("jPasswordField1");

        jCheckBox1.setText("jCheckBox1");

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("JavaComp");
        setBackground(new java.awt.Color(204, 0, 51));
        setIconImage(new javax.swing.ImageIcon("src/main/resources/images/LogoJavaComp.png").getImage());
        setResizable(false);

        editarPerfilBoton.setBackground(new java.awt.Color(255, 191, 29));
        editarPerfilBoton.setForeground(new java.awt.Color(51, 51, 51));
        editarPerfilBoton.setText("Modificar datos");
        editarPerfilBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        editarPerfilBoton.setFocusPainted(false);
        editarPerfilBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                editarPerfilBotonActionPerformed(evt);
            }
        });

        categoriaBox.setBackground(new java.awt.Color(255, 191, 29));
        categoriaBox.setForeground(new java.awt.Color(51, 51, 51));
        categoriaBox.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "Todos", "Componentes", "Ordenadores", "Móviles y telefonía", "TV, audio y foto", "Consolas y videojuegos" }));
        categoriaBox.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        categoriaBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                categoriaBoxActionPerformed(evt);
            }
        });

        categoria.setText("Categoría");

        ordenarPor.setText("Ordenar por");

        ordenarBox.setBackground(new java.awt.Color(255, 191, 29));
        ordenarBox.setForeground(new java.awt.Color(51, 51, 51));
        ordenarBox.setModel(new javax.swing.DefaultComboBoxModel<>(new String[] { "relevancia", "mayor precio", "menor precio" }));
        ordenarBox.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        ordenarBox.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                ordenarBoxActionPerformed(evt);
            }
        });

        busquedaScroll.setMaximumSize(new java.awt.Dimension(300, 32767));

        busquedaPanel.setLayout(new javax.swing.BoxLayout(busquedaPanel, javax.swing.BoxLayout.PAGE_AXIS));

        jCheckBox2.setText("jCheckBox2");
        busquedaPanel.add(jCheckBox2);

        jCheckBox3.setText("jCheckBox3");
        busquedaPanel.add(jCheckBox3);

        jCheckBox4.setText("jCheckBox4\tjpanel\txd");
        busquedaPanel.add(jCheckBox4);

        jCheckBox5.setText("jCheckBox5");
        busquedaPanel.add(jCheckBox5);

        jCheckBox6.setText("jCheckBox6");
        busquedaPanel.add(jCheckBox6);

        jCheckBox7.setText("jCheckBox7");
        busquedaPanel.add(jCheckBox7);

        busquedaScroll.setViewportView(busquedaPanel);

        carroBoton.setBackground(new java.awt.Color(255, 191, 29));
        carroBoton.setForeground(new java.awt.Color(51, 51, 51));
        carroBoton.setText("Carro y pasar a pago");
        carroBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        carroBoton.setFocusPainted(false);
        carroBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                carroBotonActionPerformed(evt);
            }
        });

        buscarBoton.setBackground(new java.awt.Color(255, 191, 29));
        buscarBoton.setForeground(new java.awt.Color(51, 51, 51));
        buscarBoton.setText("Buscar");
        buscarBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        buscarBoton.setFocusPainted(false);
        buscarBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                buscarBotonActionPerformed(evt);
            }
        });

        buscarField.setBackground(new java.awt.Color(255, 206, 29));
        buscarField.setForeground(new java.awt.Color(65, 65, 65));
        buscarField.setText("Buscar...");
        buscarField.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        buscarField.addFocusListener(new java.awt.event.FocusAdapter() {
            public void focusGained(java.awt.event.FocusEvent evt) {
                buscarFieldFocusGained(evt);
            }
            public void focusLost(java.awt.event.FocusEvent evt) {
                buscarFieldFocusLost(evt);
            }
        });

        cerrarSesionBoton.setBackground(new java.awt.Color(255, 191, 29));
        cerrarSesionBoton.setForeground(new java.awt.Color(51, 51, 51));
        cerrarSesionBoton.setText("Cerrar sesión");
        cerrarSesionBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(51, 51, 51), 3));
        cerrarSesionBoton.setFocusPainted(false);
        cerrarSesionBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                cerrarSesionBotonActionPerformed(evt);
            }
        });

        jLabel2.setFont(new java.awt.Font("Segoe UI", 0, 18)); // NOI18N
        jLabel2.setText("JavaComp");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(busquedaScroll, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addComponent(cerrarSesionBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 103, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(carroBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 143, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(buscarField, javax.swing.GroupLayout.DEFAULT_SIZE, 315, Short.MAX_VALUE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(buscarBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 54, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(iconImageLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 45, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                                .addComponent(jLabel2)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 244, Short.MAX_VALUE)))
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(categoriaBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(categoria))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(ordenarPor)
                            .addComponent(ordenarBox, 0, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(editarPerfilBoton, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(iconImageLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 45, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                        .addComponent(editarPerfilBoton, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addComponent(jLabel2, javax.swing.GroupLayout.Alignment.LEADING, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addGap(0, 0, 0)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(ordenarPor)
                    .addComponent(categoria))
                .addGap(9, 9, 9)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(ordenarBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(categoriaBox, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(buscarBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(buscarField, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(busquedaScroll, javax.swing.GroupLayout.DEFAULT_SIZE, 588, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(cerrarSesionBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(carroBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 25, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    
    //Lleva a la pestaña de editar perfil
    private void editarPerfilBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_editarPerfilBotonActionPerformed
        
        //Dependiendo de si es Particular o Empresa lleva a un frame distinto
        if (DataManager.getClienteActual() instanceof Particular) {
            ModificarDatosParticular interfModificar = new ModificarDatosParticular();
            interfModificar.setLocationRelativeTo(null);
            interfModificar.setVisible(true);
            this.setVisible(false);
            
        }
        else {
            ModificarDatosEmpresa interfModificar = new ModificarDatosEmpresa();
            interfModificar.setLocationRelativeTo(null);
            interfModificar.setVisible(true);
            this.setVisible(false);
            
        }
    }//GEN-LAST:event_editarPerfilBotonActionPerformed
    
    //Seleccion de categoria
    private void categoriaBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_categoriaBoxActionPerformed
        filtrarCategoria((String) categoriaBox.getSelectedItem(), (String) ordenarBox.getSelectedItem());
    }//GEN-LAST:event_categoriaBoxActionPerformed
    
    //Filtra la lista de DisplayProducto por categoria y la ordena por el sort que elijamos
    private void filtrarCategoria(String categoria, String sort){
        switch (sort) {
            case "relevancia":
                System.out.print("relevancia");
                if(!buscarField.getText().equals("Buscar...") && !buscarField.getText().isBlank()){
                    ArrayList filtro = DataManager.busquedaProducto(buscarField.getText());
                    DataManager.sortTitulo(filtro);
                    DataManager.sortRelevancia(filtro);
                    if (!filtro.isEmpty()){
                        DataManager.displayList(DataManager.filtrarCategoria(categoria, filtro), busquedaPanel);
                    }
                    else JOptionPane.showMessageDialog(this, "No se ha encontrado nada con los criterios de búsqueda", "Error", JOptionPane.WARNING_MESSAGE);
                }
                else if (!DataManager.filtrarCategoria(categoria, DataManager.getProductos()).isEmpty()){
                    
                    ArrayList<Producto> filtro = (ArrayList) DataManager.getProductos().clone();
                    DataManager.sortTitulo(filtro);
                    DataManager.sortRelevancia(filtro);
                    DataManager.displayList(DataManager.filtrarCategoria(categoria, filtro), busquedaPanel);
                }
                else JOptionPane.showMessageDialog(this, "No existe ningún producto en esta categoría", "Error", JOptionPane.WARNING_MESSAGE);
                break;
            case "mayor precio":
                if(!buscarField.getText().equals("Buscar...") && !buscarField.getText().isBlank()){
                    
                    ArrayList filtro = DataManager.busquedaProducto(buscarField.getText());
                    DataManager.sortTitulo(filtro);
                    DataManager.sortPrecio(false, filtro);
                    
                    if (!filtro.isEmpty()){
                        DataManager.displayList(DataManager.filtrarCategoria(categoria, filtro), busquedaPanel);
                    }
                    else JOptionPane.showMessageDialog(this, "No se ha encontrado nada con los criterios de búsqueda", "Error", JOptionPane.WARNING_MESSAGE);
                }
                else if (!DataManager.filtrarCategoria(categoria, DataManager.getProductos()).isEmpty()){
                    
                    ArrayList<Producto> filtro = (ArrayList) DataManager.getProductos().clone();
                    DataManager.sortTitulo(filtro);
                    DataManager.sortPrecio(false, filtro);
                    DataManager.displayList(DataManager.filtrarCategoria(categoria, filtro), busquedaPanel);
                }
                else JOptionPane.showMessageDialog(this, "No existe ningún producto en esta categoría", "Error", JOptionPane.WARNING_MESSAGE);
                break;
            case "menor precio":
                if(!buscarField.getText().equals("Buscar...") && !buscarField.getText().isBlank()){
                    ArrayList filtro = DataManager.busquedaProducto(buscarField.getText());
                    DataManager.sortTitulo(filtro);
                    DataManager.sortRelevancia(filtro);
                    if (!filtro.isEmpty()){
                        DataManager.displayList(DataManager.filtrarCategoria(categoria, filtro), busquedaPanel);
                    }
                    else JOptionPane.showMessageDialog(this, "No se ha encontrado nada con los criterios de búsqueda", "Error", JOptionPane.WARNING_MESSAGE);
                }
                else if (!DataManager.filtrarCategoria(categoria, DataManager.getProductos()).isEmpty()){
                    ArrayList<Producto> filtro = (ArrayList) DataManager.getProductos().clone();
                    DataManager.sortTitulo(filtro);
                    DataManager.sortPrecio(true, filtro);
                    DataManager.displayList(DataManager.filtrarCategoria(categoria, filtro), busquedaPanel);
                }
                else JOptionPane.showMessageDialog(this, "No existe ningún producto en esta categoría", "Error", JOptionPane.WARNING_MESSAGE);
                break;
        }
    }
    
    private void ordenarBoxActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_ordenarBoxActionPerformed
        // TODO add your handling code here:
        filtrarCategoria((String) categoriaBox.getSelectedItem(), (String) ordenarBox.getSelectedItem());
    }//GEN-LAST:event_ordenarBoxActionPerformed

    private void buscarBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_buscarBotonActionPerformed
        // TODO add your handling code here:
        filtrarCategoria((String) categoriaBox.getSelectedItem(), (String) ordenarBox.getSelectedItem());
    }//GEN-LAST:event_buscarBotonActionPerformed
    
    private void carroBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_carroBotonActionPerformed
        InterfCarro interf = new InterfCarro(this);
        interf.setLocationRelativeTo(null);
        interf.setVisible(true);
        this.setVisible(false);
    }//GEN-LAST:event_carroBotonActionPerformed

    private void buscarFieldFocusGained(java.awt.event.FocusEvent evt) {//GEN-FIRST:event_buscarFieldFocusGained
        if (buscarField.getText().equals("Buscar...")) buscarField.setText("");
    }//GEN-LAST:event_buscarFieldFocusGained

    private void buscarFieldFocusLost(java.awt.event.FocusEvent evt) {//GEN-FIRST:event_buscarFieldFocusLost
        if (buscarField.getText().isBlank()) buscarField.setText("Buscar...");
    }//GEN-LAST:event_buscarFieldFocusLost

    private void cerrarSesionBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_cerrarSesionBotonActionPerformed
        String[] opciones = new String[2];
        opciones[0]="Sí";
        opciones[1]="No";
        int respuesta = JOptionPane.showOptionDialog(this, "¿Estás seguro de que quieres cerrar sesión?", "Confirmar cierre de sesión", 0, JOptionPane.INFORMATION_MESSAGE, null, opciones, null);
        if (respuesta==0){
            this.setVisible(false);
            MainMenu menuPrincipal = new MainMenu();
            DataManager.setCarritoActual(new ArrayList());
            menuPrincipal.setLocationRelativeTo(null);
            menuPrincipal.setVisible(true);
        }
    }//GEN-LAST:event_cerrarSesionBotonActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(InterfCliente.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(InterfCliente.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(InterfCliente.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(InterfCliente.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new InterfCliente().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton buscarBoton;
    private javax.swing.JTextField buscarField;
    private javax.swing.JPanel busquedaPanel;
    private javax.swing.JScrollPane busquedaScroll;
    private javax.swing.JButton carroBoton;
    private javax.swing.JLabel categoria;
    private javax.swing.JComboBox<String> categoriaBox;
    private javax.swing.JButton cerrarSesionBoton;
    private javax.swing.JButton editarPerfilBoton;
    private javax.swing.JLabel iconImageLabel;
    private javax.swing.JCheckBox jCheckBox1;
    private javax.swing.JCheckBox jCheckBox2;
    private javax.swing.JCheckBox jCheckBox3;
    private javax.swing.JCheckBox jCheckBox4;
    private javax.swing.JCheckBox jCheckBox5;
    private javax.swing.JCheckBox jCheckBox6;
    private javax.swing.JCheckBox jCheckBox7;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JPasswordField jPasswordField1;
    private javax.swing.JComboBox<String> ordenarBox;
    private javax.swing.JLabel ordenarPor;
    private org.jdatepicker.UtilCalendarModel utilCalendarModel1;
    // End of variables declaration//GEN-END:variables
}
