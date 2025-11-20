from training_logic import WhistTrainer

if __name__ == "__main__":
    # Create and configure the trainer
    trainer = WhistTrainer(
        num_games=1000,
        epsilon=1.0,
        epsilon_decay=0.996,
        min_epsilon=0.001,
        array_length=13,
        gamma_values=[0.99, 0.95, 0.90, 0.85],
        save_every=500
    )
    
    # Run the training
    trainer.train()
    
    # Plot the results
    trainer.plot_results()
