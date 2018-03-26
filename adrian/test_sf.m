


SF_v = structure_function_vect( 'Blazar_LC.csv', 1 );
%SF = structure_function( 'Blazar_LC.csv', 2 );


%sum(abs(SF-SF_v))

%size(SF)
%size(SF_v)

%scatter( SF(:,1), SF(:,2), 'b' );
%hold on
scatter( SF_v(:,1), SF_v(:,2), 'g*' );