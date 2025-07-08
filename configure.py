def get_default_config(data_name):
    if data_name in ['Caltech101-20']:
        return dict(
            Prediction=dict(
                arch1=[256, 256, 128],
                arch2=[256, 256, 128],
            ),
            Autoencoder=dict(
                # arch1=[1984, 1024, 1024, 1024, 128],  # the last number is the dimension of latent representation
                # arch2=[512, 1024, 1024, 1024, 128],  # the last number in arch1 and arch2 should be the same
                arch1=[1984, 516, 516,128, 128],
                arch2=[512, 516, 516, 128, 128],

                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=128,
                seed=4,
                missing_rate=0,
                class_num=20,
                start_dual_prediction=200,
                start_tuning=350, # 100
                batch_size=256,
                epoch=1000,
                lr=1.0e-4,
                # Balanced factors for L_cd, L_pre, and L_rec
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
            ),
        )

    elif data_name in ['Scene_15']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[256, 256, 128],
                arch2=[256, 256, 128],
            ),
            Autoencoder=dict(
                # arch1=[20, 1024, 1024, 1024, 128],
                # arch2=[59, 1024, 1024, 1024, 128],
                arch1=[20, 300, 300,100, 64],
                arch2=[59, 300, 300,100, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=32,
                missing_rate=0,
                class_num=15,
                seed=8,
                start_dual_prediction=300,  # 200
                start_tuning=450,  # 250
                batch_size=256,
                epoch=800,  # 600
                lr=1.0e-3,
                # lr=0.0003,
                alpha=9,   # 9
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
            ),
        )

    elif data_name in ['NoisyMNIST']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[256, 256, 128],
                arch2=[256, 256, 128],
            ),
            Autoencoder=dict(
                # arch1=[784, 1024, 1024, 1024, 64],
                # arch2=[784, 1024, 1024, 1024, 64],
                arch1=[784, 1024, 1024, 1024, 256],
                arch2=[784, 1024, 1024, 1024, 256],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=64,
                missing_rate=0.5,
                seed=0,
                class_num=10,
                start_dual_prediction=200,
                start_tuning=450,  # 100
                epoch=600,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
            ),
        )

    elif data_name in ['LandUse_21']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[256, 256, 128],
                arch2=[256, 256, 128],
            ),
            Autoencoder=dict(
                # arch1=[59, 1024, 1024, 1024, 64],
                # arch2=[40, 1024, 1024, 1024, 64],
                arch1=[59, 500, 1000, 1024, 64],
                arch2=[40, 500, 1000, 1024, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=32,
                missing_rate=0,
                seed=3,
                class_num=21,
                start_dual_prediction=200,
                start_tuning=250,  # 100
                epoch=600,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
            ),
        )
    elif data_name in ['BDGP']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1750, 500, 500, 500, 64],
                arch2=[79, 500, 500, 500, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=64,
                missing_rate=0.5,
                class_num=5,
                seed=0,
                start_dual_prediction=400,
                start_tuning=500,
                epoch=1000,
                batch_size=256,
                lr=1.0e-3,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
            ),
        )
    elif data_name in ['Reuters_dim10']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[10, 100, 100, 100, 64],
                arch2=[10, 100, 100, 100, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=32,
                missing_rate=0.5,
                seed=0,
                start_dual_prediction=400,
                start_tuning=500,
                epoch=1000,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
                class_num=6,
            ),
        )
    elif data_name in ['CUB']:
        """The default configs."""
        return dict(
            Prediction=dict(
                arch1=[128, 256, 128],
                arch2=[128, 256, 128],
            ),
            Autoencoder=dict(
                arch1=[1024, 300, 400, 300, 64],
                arch2=[300, 300, 400, 300, 64],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
            training=dict(
                high_feature_dim=64,
                missing_rate=0.5,
                seed=0,
                start_dual_prediction=400,
                start_tuning=500,
                epoch=1000,
                batch_size=256,
                lr=1.0e-4,
                alpha=9,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
                class_num=10,
            ),
        )
    else:
        raise Exception('Undefined data_name')
