#include "layer.hpp"
#include "neural_network.hpp"
#include "spectral_image.hpp"
#include "spectral_train_set.hpp"
#include "util/const_init.hpp"
#include "util/print.hpp"
#include "util/random_init.hpp"
#include "util/timer.hpp"

#include <es_la/dense.hpp>
#include <es_la/io.hpp>

#include <iomanip>
#include <iostream>

int main()
{
	const auto image = read_image("salinas.txt");
	const auto train_set = read_train_set("salinas_train2.txt", "salinas_train2_labels.txt");

	auto network = make_neural_network(
		.03, Conv_layer(10, 20), Pooling_layer(5), Fc_layer(100), Output_layer(train_set.n_label_values));

	network.init(Random_init{.05}, train_set.spectrum_size);
	std::cout << network.info_string() << std::endl;

	// network.check_gradients(train_set.data, train_set.labels);
	// return 0;

	Timer tm;
	tm.start();
	const auto loss = network.train(train_set.data, train_set.labels, 100);
	tm.stop();

	std::cout << "Training took " << tm.sec() << " seconds" << std::endl;

	tm.start();
	const auto image_labels = network.classify(image.data);
	tm.stop();

	std::cout << "Classification took " << tm.sec() << " seconds" << std::endl;

	es_la::Matfile_writer mw("output.mat");
	mw.write("rows", image.rows);
	mw.write("cols", image.cols);
	mw.write("labels", image_labels);
	mw.write("loss_fn", loss);

	return 0;
}
