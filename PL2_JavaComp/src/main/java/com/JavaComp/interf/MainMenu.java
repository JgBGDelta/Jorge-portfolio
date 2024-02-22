/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package com.JavaComp.interf;

import javax.swing.ImageIcon;

/**
 *
 * @author Slend
 */
public class MainMenu extends javax.swing.JFrame {

    /**
     * Creates new form MainMenu
     */
    public MainMenu() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        Title = new javax.swing.JLabel();
        signInBoton = new javax.swing.JButton();
        signUpBoton = new javax.swing.JButton();
        jLabel2 = new javax.swing.JLabel();
        jLabel1 = new javax.swing.JLabel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("JavaComp");
        setIconImage(new javax.swing.ImageIcon("src/main/resources/images/LogoJavaComp.png").getImage());

        Title.setFont(new java.awt.Font("Sitka Display", 1, 48)); // NOI18N
        Title.setForeground(new java.awt.Color(189, 119, 22));
        Title.setText("Tienda JavaComp");

        signInBoton.setBackground(new java.awt.Color(255, 191, 29));
        signInBoton.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        signInBoton.setForeground(new java.awt.Color(0, 0, 0));
        signInBoton.setText("Sign In");
        signInBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0), 3));
        signInBoton.setFocusPainted(false);
        signInBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                signInBotonActionPerformed(evt);
            }
        });

        signUpBoton.setBackground(new java.awt.Color(255, 191, 29));
        signUpBoton.setFont(new java.awt.Font("Segoe UI", 1, 18)); // NOI18N
        signUpBoton.setForeground(new java.awt.Color(0, 0, 0));
        signUpBoton.setText("Sign Up");
        signUpBoton.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0), 3));
        signUpBoton.setFocusPainted(false);
        signUpBoton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                signUpBotonActionPerformed(evt);
            }
        });

        jLabel2.setIcon(new ImageIcon("src/main/resources/images/LogoJavaComp.png"));

        jLabel1.setFont(new java.awt.Font("Sitka Display", 0, 24)); // NOI18N
        jLabel1.setText("Consigue tus productos al mejor precio");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(layout.createSequentialGroup()
                                .addGap(0, 0, Short.MAX_VALUE)
                                .addComponent(signUpBoton)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(signInBoton, javax.swing.GroupLayout.PREFERRED_SIZE, 68, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(Title)
                                .addGap(0, 0, Short.MAX_VALUE))))
                    .addGroup(layout.createSequentialGroup()
                        .addContainerGap(268, Short.MAX_VALUE)
                        .addComponent(jLabel1)
                        .addGap(0, 267, Short.MAX_VALUE)))
                .addContainerGap())
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addComponent(jLabel2)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(signInBoton)
                    .addComponent(signUpBoton))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 15, Short.MAX_VALUE)
                .addComponent(jLabel2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(Title)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel1)
                .addContainerGap(10, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents
    //Redirige a la ventana del "Sign in"
    private void signInBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_signInBotonActionPerformed
        this.setVisible(false);
        SignIn signIn = new SignIn();
        signIn.prevFrame = this;
        signIn.setLocationRelativeTo(null);
        signIn.setVisible(true);
    }//GEN-LAST:event_signInBotonActionPerformed
//Redirige a la ventana del "Sign up"
    private void signUpBotonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_signUpBotonActionPerformed
        this.setVisible(false);
        SignUp signUp = new SignUp();
        signUp.prevFrame = this;
        signUp.setLocationRelativeTo(null);
        signUp.setVisible(true);
    }//GEN-LAST:event_signUpBotonActionPerformed

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
                if ("Windows".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(MainMenu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(MainMenu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(MainMenu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(MainMenu.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new MainMenu().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel Title;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JButton signInBoton;
    private javax.swing.JButton signUpBoton;
    // End of variables declaration//GEN-END:variables
}
